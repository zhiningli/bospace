from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood

from botorch.models import SingleTaskGP
from botorch.acquisition import UpperConfidenceBound
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.fit import fit_gpytorch_mll_torch
from botorch.optim.optimize import optimize_acqf_discrete

import torch
from tqdm import tqdm
from itertools import product
from src.middleware import ComponentStore
import logging
from typing import Callable

logger = logging.getLogger("service")

class BO_Service:

    def __init__(self, component_store: ComponentStore):
        self.objective_func: Callable = None
        self.search_space: dict[str, list[float]] = None
        self.bounds: torch.Tensor = None
        self._store: ComponentStore = component_store

    def optimise(self,
                search_space: dict[str, list[float]],
                sample_per_batch: int = 1,
                n_iter: int = 25, 
                initial_points: int = 20) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] :
        r"""
        Optimize the hyperparameters using Bayesian Optimization.

        Params:
        search_space: A dict object with string being the search space name and value being the search space range
        sample_per_patch: int specifying sample per batch
        n_iter: iteration count
        initial_points: number of initial point to sample

        Returns:
        
        Tuple containing:
        Accuracies: torch.Torch recording the accuracies along the way
        Best_y: torch scalar Tensor showing the best way
        Best_candidate: torch.Tensor showing the minimum found
        """
        self.search_space = search_space
        self.bounds = torch.Tensor([
                                    [0, 0, 0, 0],
                                    [len(self.search_space['learning_rate'])-1, 
                                     len(self.search_space['momentum'])-1, 
                                     len(self.search_space['weight_decay'])-1, 
                                     len(self.search_space['num_epochs'])-1]
                                    ])

        if not callable(self._store.objective_func):
            logger.error("Unable to execute the objective function, potentially it is not instantiated by the component store")
            raise ValueError("Unable to execute the objective function, check if it is instantiated using self._store.instantiate_all_classes()")
        try:
            logging.info("Initial check completed, starting bayesian optimisation")
            accuracies, best_y, best_candidate = self._run_bayesian_optimisation(n_iter=n_iter, initial_points = initial_points,sample_per_batch= sample_per_batch)
            logging.info("Bayesian optimisation completed")
            return accuracies, best_y, best_candidate
        except Exception as e:
            logging.error(f"Bayesian optimisation failed {e}", exc_info=True)
            raise RuntimeError(f"Error while running bayesian optimisation {e}")

    def _botorch_objective(self, x: torch.Tensor) -> torch.Tensor:
        """
        A thin wrapper to map input tensor to kwargs to objective function
        """
        np_params = x.detach().cpu().numpy().squeeze()
        params = {
            "learning_rate": self.search_space["learning_rate"][int(np_params[0])],
            "momentum": self.search_space["momentum"][int(np_params[1])],
            "weight_decay": self.search_space["weight_decay"][int(np_params[2])],
            "num_epochs": self.search_space["num_epochs"][int(np_params[3])]
        }

        return torch.tensor(self._store.objective_func(**params), dtype=torch.float64)

    def _run_bayesian_optimisation(self, 
                                    n_iter: int,
                                    initial_points: int,
                                    sample_per_batch: int,
                                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Commencing bayesian optimsation on device: {device}")

        bounds = self.bounds.to(device)

        train_x = draw_sobol_samples(bounds=bounds, n=initial_points, q=sample_per_batch).squeeze(1).to(device, torch.float64)
        train_y = torch.zeros((len(train_x), 1), dtype=torch.float64, device=device)

        logger.debug("Sampling initial random samples")
        for i, x in enumerate(tqdm(train_x, desc="Initial sampling")):
            train_y[i] = self._botorch_objective(x).item()


        normalised_train_x = self._normalize_to_unit_cube(train_x, bounds)
        choices = torch.tensor(list(product(*[range(len(self.search_space[dim])) for dim in self.search_space])), dtype=torch.float64, device=device)
        normalized_choices = choices / torch.tensor(
            [len(self.search_space[dim]) - 1 for dim in self.search_space],
            dtype=torch.float64, device=device
        )


        likelihood = GaussianLikelihood(noise_constraint=GreaterThan(1e-6)).to(torch.float64, deice=cuda)
        gp = (SingleTaskGP(
            train_X = normalised_train_x,
            train_Y= train_y,
            likelihood=likelihood,
        ).to(torch.float64, device = device))

        mll = ExactMarginalLogLikelihood(likelihood, gp).to(torch.float64, device = device) 

        logger.debug("Fitting gaussian process surrogate to objective function...")
        fit_gpytorch_mll_torch(mll)
        acq_function = UpperConfidenceBound(model = gp, beta = 2).to(device)

        best_candidate = None
        best_y = float('-inf')
        accuracies = []

        logger.debug("Running bayesian optimisation")
        with tqdm(total=n_iter, desc="Bayesian Optimization Progress", unit="iter") as pbar:
            for i in range(n_iter):
                candidate, _ = optimize_acqf_discrete(
                    acq_function=acq_function,
                    q=1,
                    choices=normalized_choices, 
                    max_batch_size=2048,
                    unique=True
                )
                
                candidate = self._denormalize_from_unit_cube(candidate, bounds)
                
                new_y = self._botorch_objective(candidate).view(1, 1).to(device)
                new_y_value = new_y.item()

                if new_y_value >= best_y:
                    best_y = new_y_value
                    best_candidate = candidate

                candidate = self._normalize_to_unit_cube(candidate, bounds)

                accuracies.append(new_y_value)

                normalised_train_x = torch.cat([normalised_train_x, candidate.view(1, -1)]).to(device)
                train_y = torch.cat([train_y.view(-1, 1), new_y], dim=0).view(-1)

                gp.set_train_data(inputs=normalised_train_x, targets=train_y, strict=False)
                acq_function = UpperConfidenceBound(model = gp, beta = 2).to(device)

                pbar.set_postfix({"Best Y": best_y})
                pbar.update(1)
        return accuracies, best_y, best_candidate
    
    def _normalize_to_unit_cube(self, data: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
        lower_bounds = bounds[0].to(data.device)
        upper_bounds = bounds[1].to(data.device)
        normalized = (data - lower_bounds) / (upper_bounds - lower_bounds)
        return normalized.to(data.device)

    def _denormalize_from_unit_cube(self, data: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
        lower_bounds = bounds[0].to(data.device)
        upper_bounds = bounds[1].to(data.device)
        denormalized = data * (upper_bounds - lower_bounds) + lower_bounds
        return denormalized.to(data.device)
