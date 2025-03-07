from src.middleware import ComponentStore
from src.service.embeddings import Model_Embedder, Dataset_Embedder
from src.model import ModelSimilarityModel, DatasetSimilarityModel
from src.database.crud import ModelRepository, DatasetRepository, ScriptRepository, SimilarityRepository
from src.utils import extract_model_source_code
from src.assets.search_space import sgd_search_space

import os
from datetime import datetime
import joblib
import xgboost as xgb
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error

class SimilarityInferenceService:

    def __init__(self):
        self.store = ComponentStore()
        self.model_embedder = Model_Embedder()
        self.dataset_embedder = Dataset_Embedder()
        self.search_space = sgd_search_space
        self.model_predictor = ModelSimilarityModel()
        self.dataset_predictor = DatasetSimilarityModel()
        
        self._instantiate_ML_models()
        
    def suggest_search_space(self, code_str: str, num_similar_model: int = 5, num_similar_dataset: int = 5) -> dict[str, dict[str, float]]:
        

        self.store.code_string = code_str
        self.store.instantiate_code_classes()

        model_source_code = extract_model_source_code(code_str)
        dataset_instance = self.store.dataset_instance

        top_k_similar_models = self.compute_top_k_model_similarities(model_source_code, k=num_similar_model)
        print("top_k_models", top_k_similar_models)
        top_k_similar_datasets = self.compute_top_k_dataset_similarities(dataset_instance=dataset_instance, k =num_similar_dataset)
        print("top_k_datasets", top_k_similar_datasets)
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

    def compute_top_k_model_similarities(self, model_source_code, k):
        res = []
        
        target_model_embedding = self.model_embedder.get_embedding(model_source_code)

        models = ModelRepository.get_all_models_with_feature_vector_only()

        for model_object in models:
            source_model_idx = model_object.model_idx
            source_model_embedding = model_object.feature_vector
            if source_model_idx > 30:
                continue
            features = np.array(target_model_embedding + source_model_embedding).reshape(1, -1)
            score = self.model_predictor.predict(features)
            res.append((source_model_idx, score))
        
        res.sort(key= lambda x:x[1], reverse=True)
        return [res[i][0] for i in range(k)]

    def compute_top_k_dataset_similarities(self, dataset_instance, k):
        res = []
        self.dataset_embedder.set_data(dataset_instance)
        target_meta_features = self.dataset_embedder.extract_meta_features().tolist()
        datasets = DatasetRepository.get_all_datasets_with_meta_features()

        for dataset_object in datasets:
            source_dataset_idx = dataset_object.dataset_idx
            if source_dataset_idx > 30:
                continue
            source_dataset_meta_features = dataset_object.meta_features
            features = np.array(target_meta_features + source_dataset_meta_features).reshape(1, -1)

            score = self.dataset_predictor.predict(features)
            
            res.append((source_dataset_idx, score))
        
        res.sort(key= lambda x:x[1], reverse=True)
        return [res[i][0] for i in range(k)]

    def _instantiate_ML_models(self):
        load_dotenv()
        model_similarity_file = os.path.join(os.getenv("MODEL_RANK_PREDICTION_MODEL_PATH", "./"), "model_similarity.pkl")
        self.model_predictor.load_model(model_similarity_file)

        dataset_similarity_file = os.path.join(os.getenv("DATASET_RANK_PREDICTION_MODEL_PATH", "./"), "dataset_similarity.pkl")
        self.dataset_predictor.load_model(dataset_similarity_file)