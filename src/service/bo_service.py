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
import bisect
from itertools import product
from src.middleware import ComponentStore
from src.assets import sgd_search_space
import logging
from typing import Callable

logger = logging.getLogger("service")

class BOService:

    def __init__(self, component_store: ComponentStore= None):
        self.objective_func: Callable = None
        self.search_space: dict[str, list[float]] = None
        self.bounds: torch.Tensor = None
        if not component_store:
            self._store = ComponentStore()
        else:
            self._store: ComponentStore = component_store
        


    def load_constrained_search_space(self, search_space: dict[str, dict[str, float]]):
        """Transform the constrained search space from similarity_inference_service to a search space understandable by BO."""

        def find_nearest_index(search_list, value, lower):
            """Find the index of the nearest value using binary search."""
            if lower:
                idx = bisect.bisect_left(search_list, value)
            else:
                idx = bisect.bisect_right(search_list, value)
            if idx == 0:
                return 0
            if idx == len(search_list):
                return len(search_list) - 1
            
            if abs(search_list[idx] - value) < abs(search_list[idx - 1] - value):
                return idx
            else:
                return idx - 1

        lower = search_space["lower_bound"]
        upper = search_space["upper_bound"]

        # Find nearest indices using binary search
        lower_learning_rate_idx = find_nearest_index(sgd_search_space["learning_rate"], lower["learning_rate"], True)
        upper_learning_rate_idx = find_nearest_index(sgd_search_space["learning_rate"], upper["learning_rate"], False)

        lower_momentum_idx = find_nearest_index(sgd_search_space["momentum"], lower["momentum"], lower = True)
        upper_momentum_idx = find_nearest_index(sgd_search_space["momentum"], upper["momentum"], lower = False)

        lower_num_epochs_idx = find_nearest_index(sgd_search_space["num_epochs"], lower["num_epochs"], True)
        upper_num_epochs_idx = find_nearest_index(sgd_search_space["num_epochs"], upper["num_epochs"], False)

        lower_weight_decay_idx = find_nearest_index(sgd_search_space["weight_decay"], lower["weight_decay"], True)
        upper_weight_decay_idx = find_nearest_index(sgd_search_space["weight_decay"], upper["weight_decay"], False)

        # Construct the constrained search space
        constrained_search_space = {
            "learning_rate": sgd_search_space["learning_rate"][lower_learning_rate_idx:upper_learning_rate_idx + 1],
            "momentum": sgd_search_space["momentum"][lower_momentum_idx:upper_momentum_idx + 1],
            "num_epochs": sgd_search_space["num_epochs"][lower_num_epochs_idx:upper_num_epochs_idx + 1],
            "weight_decay": sgd_search_space["weight_decay"][lower_weight_decay_idx:upper_weight_decay_idx + 1]
        }

        return constrained_search_space


    def optimise(self,
                code_str: str,
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
        if code_str:
            self._store.code_string = code_str
            self._store.instantiate_code_classes()

        if not callable(self._store.objective_func):
            logger.error("Unable to execute the objective function, potentially it is not instantiated by the component store")
            raise ValueError("Unable to execute the objective function, check if it is instantiated using self._store.instantiate_all_classes()")
        try:
            logger.info("Initial check completed, starting bayesian optimisation")
            accuracies, best_y, best_candidate = self._run_bayesian_optimisation(n_iter=n_iter, initial_points = initial_points,sample_per_batch= sample_per_batch)
            logger.info("Bayesian optimisation completed")
            return accuracies, best_y, best_candidate
        except Exception as e:
            logger.error(f"Bayesian optimisation failed {e}", exc_info=True)
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

        return torch.tensor(self._store.objective_func(**params), dtype=torch.float64, device="cuda" if torch.cuda.is_available() else "cpu")

    def _run_bayesian_optimisation(self, 
                                    n_iter: int,
                                    initial_points: int,
                                    sample_per_batch: int,
                                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Commencing bayesian optimsation on device: {device}")
        logger.info(f"Commencing bayesian optimsation on device: {device}")

        bounds = self.bounds.to(device)

        train_x = draw_sobol_samples(bounds=bounds, n=initial_points, q=sample_per_batch).squeeze(1).to(device, torch.float64)
        train_y = torch.zeros((len(train_x), 1), dtype=torch.float64, device=device)
        print(f"train_x is on device: {train_x.device}")
        print(f"train_y is on device: {train_y.device}")
        logger.debug("Sampling initial random samples")
        for i, x in enumerate(tqdm(train_x, desc="Initial sampling")):
            train_y[i] = self._botorch_objective(x).item()


        normalised_train_x = self._normalize_to_unit_cube(train_x, bounds)
        choices = torch.tensor(list(product(*[range(len(self.search_space[dim])) for dim in self.search_space])), dtype=torch.float64, device=device)
        normalized_choices = choices / torch.tensor(
            [len(self.search_space[dim]) - 1 for dim in self.search_space],
            dtype=torch.float64, device=device
        )


        likelihood = GaussianLikelihood(noise_constraint=GreaterThan(1e-6)).to(torch.float64).to(device)
        gp = (SingleTaskGP(
            train_X = normalised_train_x,
            train_Y= train_y,
            likelihood=likelihood,
        ).to(torch.float64)).to(device=device)
        print(f"Gaussian Process model is on device: {next(gp.parameters()).device}")

        mll = ExactMarginalLogLikelihood(likelihood, gp).to(torch.float64).to(device=device) 

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
                                                                                                                                                          