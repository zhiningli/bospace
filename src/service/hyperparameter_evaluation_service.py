""" This service is for evaluating a model or a dataset's response to a set of predefined hyperparameter configurations. 
This information is crucial for evaluating the rank correlation between two datasets or two models.

Results for saved to the HPO-evaluation table.
"""

from src.database.crud import ModelRepository
from src.database.crud import DatasetRepository
from src.database.crud import HPEvaluationRepository
from src.database.crud import ScriptRepository
from src.middleware.component_store import ComponentStore
from src.assets.train_via_sgd import code_str
import numpy as np

from scipy.stats import qmc

class HPEvalutaionService:

    def __init__(self):

        self.model_benchmarks_index = []
        self.dataset_benchmarks_index = [4, 8, 12, 17, 20, 24, 34, 40, 44, 48]
        self.manual_seed = 42
        self.search_space = {
            'learning_rate': np.logspace(-5, -1, num=50).tolist(),  # Logarithmically spaced values
            'momentum': [0.01 * x for x in range(100)],  # Linear space
            'weight_decay': [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'num_epochs': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90],}
        self.store = ComponentStore()
        self.samples = self._generate_hp_samples()

    def _generate_hp_samples(self) -> np.ndarray:
        bounds = [ [0, len(self.search_space['learning_rate'])-1], [0,len(self.search_space['momentum'])-1 ], [0, len(self.search_space['weight_decay'])-1], [0,len(self.search_space['num_epochs'])-1]]          
        n_samples = 15
        samples = qmc.Sobol(d=len(bounds), seed=self.manual_seed).random(n=n_samples)
        scaled_samples = qmc.scale(samples, [b[0] for b in bounds], [b[1] for b in bounds])

        discrete_samples = np.rint(scaled_samples).astype(int)
        return discrete_samples 

    def run_hp_evaluations_for_all_models(self):

        dataset_benchmarks = [DatasetRepository.get_dataset(i) for i in self.dataset_benchmarks_index]
        models = ModelRepository.get_all_models()
        for dataset_benchmark in dataset_benchmarks:
                
            dataset_idx = dataset_benchmark.dataset_idx
            dataset_code = dataset_benchmark.code
            dataset_input_size = dataset_benchmark.input_size
            dataset_num_classes = dataset_benchmark.num_classes

            for model in models:
                model_idx = model.model_idx
                model_code = model.code
                print(f"Model idx {model_idx} Dataset idx {dataset_idx} running")
                if HPEvaluationRepository.exists_hp_evaluation(model_idx=model_idx, dataset_idx=dataset_idx):
                    print(f"HPO evalution for model {model_idx}, dataset {dataset_idx} already exists in table, skipping ")
                    continue

                script = ScriptRepository.get_script_by_model_and_dataset_idx(model_idx= model_idx, dataset_idx= dataset_idx)


                current_code_str = code_str.format(dataset=dataset_code, model = model_code, input_size = dataset_input_size, num_classes = dataset_num_classes)
                
                self.store.code_string = current_code_str
                self.store.instantiate_code_classes()
                train_y = []
                for sample in self.samples:
                    kwargs = {
                        "learning_rate": self.search_space["learning_rate"][sample[0]],
                        "momentum": self.search_space["momentum"][sample[1]],
                        "weight_decay": self.search_space["weight_decay"][sample[2]],
                        "num_epochs": self.search_space["num_epochs"][sample[3]]
                    }

                    kwargs["accuracy"] = self.store.objective_func(**kwargs)
                    print(kwargs)
                    train_y.append(kwargs)
                    
                HPEvaluationRepository.create_hp_evaluation(
                    model_idx=model_idx,
                    dataset_idx=dataset_idx,
                    results=train_y
                )

                ScriptRepository.update_script_code(
                    script_idx=script.script_idx,
                    script_code = current_code_str
                )

