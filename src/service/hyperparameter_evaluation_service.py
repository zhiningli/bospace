""" This service is for evaluating a model or a dataset's response to a set of predefined hyperparameter configurations. 
This information is crucial for evaluating the rank correlation between two datasets or two models.

Results for saved to the HPO-evaluation table.
"""

from src.database.crud import ModelRepository
from src.database.crud import DatasetRepository
from src.database.crud import HPEvaluationRepository
from src.database.crud import ScriptRepository
from src.middleware.component_store import ComponentStore
from src.assets.search_space import sgd_search_space
from src.assets.train_via_sgd import code_str
import numpy as np
import logging
from scipy.stats import qmc

logger = logging.getLogger("service")


class HPEvaluationService:
    def __init__(self):
        logger.debug("Instantiating HPEvaluationService")

        self.model_benchmarks_index = [3, 5, 20, 1, 77, 99, 53, 30, 32, 38, 41, 10]
        self.dataset_benchmarks_index = [4, 9, 12, 17, 20, 24, 34, 40, 44, 48]
        self.manual_seed = 42
        self.search_space = sgd_search_space
        self.store = ComponentStore()
        self.samples = self._generate_hp_samples()

        logger.info("HPEvaluationService instantiated successfully")

    def _generate_hp_samples(self) -> np.ndarray:
        """Generate Sobol hyperparameter samples."""
        logger.debug("Generating hyperparameter samples using Sobol sequence")
        try:
            bounds = [
                [0, len(self.search_space["learning_rate"]) - 1],
                [0, len(self.search_space["momentum"]) - 1],
                [0, len(self.search_space["weight_decay"]) - 1],
                [0, len(self.search_space["num_epochs"]) - 1],
            ]
            n_samples = 15
            samples = qmc.Sobol(d=len(bounds), seed=self.manual_seed).random(n=n_samples)
            scaled_samples = qmc.scale(samples, [b[0] for b in bounds], [b[1] for b in bounds])
            discrete_samples = np.rint(scaled_samples).astype(int)
            logger.info(f"{n_samples} hyperparameter samples generated successfully")
            return discrete_samples
        except Exception as e:
            logger.error(f"Failed to generate hyperparameter samples: {e}")
            raise

    def run_hp_evaluations_for_all_models(self):
        """Evaluate all models against benchmark datasets."""
        logger.info("Starting hyperparameter evaluations for all models.")
        dataset_benchmarks = [DatasetRepository.get_dataset(dataset_idx=dataset_idx) for dataset_idx in self.dataset_benchmarks_index]
        models = ModelRepository.get_all_models()

        for dataset_benchmark in dataset_benchmarks:
            dataset_idx = dataset_benchmark.dataset_idx
            dataset_code = dataset_benchmark.code
            dataset_input_size = dataset_benchmark.input_size
            dataset_num_classes = dataset_benchmark.num_classes

            for model in models:
                model_idx = model.model_idx
                model_code = model.code

                logger.debug(f"Evaluating Model ID {model_idx} with Dataset ID {dataset_idx}")

                if HPEvaluationRepository.exists_hp_evaluation(model_idx=model_idx, dataset_idx=dataset_idx):
                    logger.warning(f"Evaluation for model {model_idx}, dataset {dataset_idx} already exists. Skipping.")
                    continue

                try:
                    script = ScriptRepository.get_script_by_model_and_dataset_idx(model_idx, dataset_idx)
                    current_code_str = code_str.format(
                        dataset=dataset_code,
                        model=model_code,
                        input_size=dataset_input_size,
                        num_classes=dataset_num_classes,
                    )

                    self.store.code_string = current_code_str
                    self.store.instantiate_code_classes()

                    train_y = []
                    for sample in self.samples:
                        kwargs = {
                            "learning_rate": self.search_space["learning_rate"][sample[0]],
                            "momentum": self.search_space["momentum"][sample[1]],
                            "weight_decay": self.search_space["weight_decay"][sample[2]],
                            "num_epochs": self.search_space["num_epochs"][sample[3]],
                        }

                        kwargs["accuracy"] = self.store.objective_func(**kwargs)
                        train_y.append(kwargs)

                    # Store results
                    HPEvaluationRepository.create_hp_evaluation(
                        model_idx=model_idx,
                        dataset_idx=dataset_idx,
                        results=train_y,
                    )

                    if script:
                        ScriptRepository.update_script_code(
                            script_idx=script.script_idx,
                            script_code=current_code_str,
                    )

                    logger.info(f"Successfully evaluated model {model_idx} with dataset {dataset_idx}.")

                except Exception as e:
                    logger.error(f"Failed to evaluate model {model_idx} with dataset {dataset_idx}: {e}")

        logger.info("Completed hyperparameter evaluations for all models.")

    def run_hp_evaluations_for_all_datasets(self):
        """Evaluate all datasets against benchmark models."""
        logger.info("Starting hyperparameter evaluations for all datasets.")
        model_benchmarks = [ModelRepository.get_model(model_idx=model_idx) for model_idx in self.model_benchmarks_index]
        datasets = DatasetRepository.get_all_dataset()

        for model_benchmark in model_benchmarks:
            model_idx = model_benchmark.model_idx
            model_code = model_benchmark.code

            for dataset in datasets:
                dataset_idx = dataset.dataset_idx
                dataset_code = dataset.code
                dataset_input_size = dataset.input_size
                dataset_num_classes = dataset.num_classes

                logger.debug(f"Evaluating Dataset ID {dataset_idx} with Model ID {model_idx}")

                if HPEvaluationRepository.exists_hp_evaluation(model_idx=model_idx, dataset_idx=dataset_idx):
                    logger.warning(f"Evaluation for model {model_idx}, dataset {dataset_idx} already exists. Skipping.")
                    continue

                try:
                    current_code_str = code_str.format(
                        dataset=dataset_code,
                        model=model_code,
                        input_size=dataset_input_size,
                        num_classes=dataset_num_classes,
                    )

                    self.store.code_string = current_code_str
                    self.store.instantiate_code_classes()

                    train_y = []
                    for sample in self.samples:
                        kwargs = {
                            "learning_rate": self.search_space["learning_rate"][sample[0]],
                            "momentum": self.search_space["momentum"][sample[1]],
                            "weight_decay": self.search_space["weight_decay"][sample[2]],
                            "num_epochs": self.search_space["num_epochs"][sample[3]],
                        }

                        kwargs["accuracy"] = self.store.objective_func(**kwargs)
                        train_y.append(kwargs)

                    # Store results
                    HPEvaluationRepository.create_hp_evaluation(
                        model_idx=model_idx,
                        dataset_idx=dataset_idx,
                        results=train_y,
                    )

                    script = ScriptRepository.get_script_by_model_and_dataset_idx(model_idx, dataset_idx)
                    if script:
                        ScriptRepository.update_script_code(
                            script_idx=script.script_idx,
                            script_code=current_code_str,
                        )

                    logger.info(f"Successfully evaluated dataset {dataset_idx} with model {model_idx}.")

                except Exception as e:
                    logger.error(f"Failed to evaluate dataset {dataset_idx} with model {model_idx}: {e}")

        logger.info("Completed hyperparameter evaluations for all datasets.")
