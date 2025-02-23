from src.middleware import ComponentStore
from src.preprocessing import Tokeniser
from src.service.embeddings import Model_Embedder, Dataset_Embedder
from src.database.crud import ModelRepository, DatasetRepository, ScriptRepository
from src.utils import extract_model_source_code
from src.assets.search_space import sgd_search_space

class SimilarityInferenceService:

    def __init__(self):
        self.store = ComponentStore()
        self.model_embedder = Model_Embedder()
        self.dataset_embedder = Dataset_Embedder()
        self.search_space = sgd_search_space
        
    def suggest_search_space(self, code_str: str) -> dict[str, dict[str, float]]:
        

        self.store.code_string = code_str
        self.store.instantiate_code_classes()

        model_source_code = extract_model_source_code(code_str)
        dataset_instance = self.store.dataset_instance

        top_k_similar_models = self.compute_top_k_model_similarities(model_source_code)
        top_k_similar_datasets = self.compute_top_k_dataset_similarities(dataset_instance=dataset_instance)

        result = {
            "lower_bound": {
                "learning_rate": float("inf"),
                "weight_decay": float("inf"),
                "num_epochs": float("inf"),
                "momentum": float("inf")
            },
            "upper_bound": {
                "learning_rate": float("-inf"),
                "weight_decay": float("-inf"),
                "num_epochs": float("-inf"),
                "momentum": float("-inf")
            }
        }

        # Constructing a compact search space using a pretty brute force method
        for model_idx in top_k_similar_models:
            for dataset_idx in top_k_similar_datasets:
                source_script_candidate = ScriptRepository.get_script_by_model_and_dataset_idx(model_idx=model_idx, dataset_idx=dataset_idx)
                source_sgd_best_performing_candidate = source_script_candidate.sgd_best_performing_configuration
                
                source_learning_rate = source_sgd_best_performing_candidate["learning_rate"]
                result["lower_bound"]["learning_rate"] = min(source_learning_rate, result["lower_bound"]["learning_rate"])
                result["upper_bound"]["learning_rate"] = max(source_learning_rate, result["upper_bound"]["learning_rate"])

                source_weight_decay = source_sgd_best_performing_candidate["weight_decay"]
                result["lower_bound"]["weight_decay"] = min(source_weight_decay, result["lower_bound"]["weight_decay"])
                result["upper_bound"]["weight_decay"] = max(source_weight_decay, result["upper_bound"]["weight_decay"])

                source_num_epochs = source_sgd_best_performing_candidate["num_epochs"]
                result["lower_bound"]["num_epochs"] = min(source_num_epochs, result["lower_bound"]["num_epochs"])
                result["upper_bound"]["num_epochs"] = max(source_num_epochs, result["upper_bound"]["num_epochs"])

                source_momentum = source_sgd_best_performing_candidate["momentum"]
                result["lower_bound"]["momentum"] = min(source_momentum, result["lower_bound"]["momentum"])
                result["upper_bound"]["momentum"] = max(source_momentum, result["upper_bound"]["momentum"])

        return result

    def compute_top_k_model_similarities(self, model_source_code):
        res = []
        tokeniser = Tokeniser()
        tokens = tokeniser(model_source_code)
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        
        target_model_embedding = self.model_embedder.get_embedding(input_ids=input_ids, attention_mask=attention_mask)

        # Can be optimised in futher to only fetch index and embeddings
        models = ModelRepository.get_all_models()

        for model_object in models:
            source_model_idx = model_object.model_idx
            source_model_embeddings = model_object.feature_vector

            # TODO, load the XGBoost model for infering an similarity score
            break

        pass

    def compute_top_k_dataset_similarities(self, dataset_instance):

        target_meta_features = self.dataset_embedder.extract_meta_features(dataset_instance)

        datasets = DatasetRepository.get_all_dataset()

        for dataset_object in datasets:
            source_dataset_idx = dataset_object.dataset_idx
            source_dataset_meta_features = dataset_object.meta_features


            # TODO, load the RGRegressor model for infering an similarity score
            break

        pass


