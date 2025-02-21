"""
This service handles preparation of rank results, training XGBoost model for model rank prediction and training RFRegressor model for dataset rank prediction
"""
from src.database.crud import DatasetRepository, SimilarityRepository, HPEvaluationRepository, ModelRepository
import itertools
import numpy as np
from dotenv import load_dotenv
from collections import defaultdict

class SimilarityTrainingService:

    def __init__(self):
        load_dotenv()
        self.dataset_similarity_model = None
        self.model_similarity_model = None

    def prepare_data_for_rank_training(self):
        
        # Prepare rank data by computing the kendall tau rank correlation
        datasets = DatasetRepository.get_all_dataset()
        datasets_hpo_performances = HPEvaluationRepository.get_average_accuracy_per_JSON_array_index_group_by_dataset()
        performances = defaultdict(list)
        for dataset_performance in datasets_hpo_performances:
            performances[dataset_performance[0]].append(dataset_performance[1])
        print(performances)

        for one_dataset in datasets:
            one_dataset_idx = one_dataset.dataset_idx
            one_dataset_meta_features = one_dataset.meta_features
            for another_dataset in datasets:
                another_dataset_idx = another_dataset.dataset_idx
                another_dataset_meta_features = another_dataset.meta_features

                if SimilarityRepository.exists_similarity(object_type="dataset", object_1_idx=one_dataset_idx, object_2_idx=another_dataset_idx):
                    continue
                
                one_dataset_performance = performances[one_dataset_idx]
                another_dataset_performance = performances[another_dataset_idx]

                rank_score = self._kendall_tau_rank_correlation(one_dataset_performance, another_dataset_performance)

                SimilarityRepository.create_similarity(
                    object_type="dataset",
                    object_1=one_dataset_meta_features, object_1_idx=one_dataset_idx,
                    object_2_idx=another_dataset_idx, object_2=another_dataset_meta_features,
                    similarity=rank_score 
                )
           
        # Prepare rank data by computing the kendall tau rank correlation
        models = ModelRepository.get_all_models()
        models_hpo_performances = HPEvaluationRepository.get_average_accuracy_per_JSON_array_index_group_by_model()
        performances = defaultdict(list)
        for model_performance in models_hpo_performances:
            performances[model_performance[0]].append(model_performance[1])

        for one_model in models:
            one_model_idx = one_model.model_idx
            one_model_feature_vector = one_model.feature_vector
            for another_model in models:
                another_model_idx = another_model.model_idx
                another_model_feature_vector = another_model.feature_vector

                if SimilarityRepository.exists_similarity(object_type="model", object_1_idx=one_model_idx, object_2_idx=another_model_idx):
                    continue
                
                one_model_performance = performances[one_model_idx]
                another_model_performance = performances[another_model_idx]

                rank_score = self._kendall_tau_rank_correlation(one_model_performance, another_model_performance)

                SimilarityRepository.create_similarity(
                    object_type="model",
                    object_1=one_model_feature_vector, object_1_idx=one_model_idx,
                    object_2_idx=another_model_idx, object_2=another_model_feature_vector,
                    similarity=rank_score 
                )


    def training_RFRegressor_for_dataset_rank_prediction(self):
        pass

    def training_XGBoost_for_model_rank_prediction(self):
        pass




    def _kendall_tau_rank_correlation(self, d1_scores: list | np.ndarray, d2_scores: list | np.ndarray) -> float:

        num_configs = len(d1_scores)

        d1_scores = np.array(d1_scores)
        d2_scores = np.array(d2_scores)

        total_pairs = num_configs * (num_configs - 1) // 2

        concordant_discordant_count = 0

        for (i, j) in itertools.combinations(range(num_configs), 2):
            d1_relation = np.sign(d1_scores[i] - d1_scores[j])
            d2_relation = np.sign(d2_scores[i] - d2_scores[j])

            if d1_relation == d2_relation:
                concordant_discordant_count += 1
            else:
                concordant_discordant_count -= 1

        ktrc = concordant_discordant_count / total_pairs

        return ktrc