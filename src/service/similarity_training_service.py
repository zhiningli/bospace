"""
This service handles preparation of rank results, training XGBoost model for model rank prediction and training RFRegressor model for dataset rank prediction
"""
from src.database.crud import DatasetRepository, SimilarityRepository, ModelRepository, EvaluationMaterialisedView
import numpy as np
import os
from dotenv import load_dotenv
from collections import defaultdict
import joblib
from scipy.stats import spearmanr

import logging
from datetime import datetime
from src.model import DatasetSimilarityModel, ModelSimilarityModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import json

logger = logging.getLogger("service")


class SimilarityTrainingService:

    def __init__(self):
        load_dotenv()
        self.dataset_similarity_model = None
        self.model_similarity_model = None

    def prepare_data_for_rank_training(self):
        try:
            logger.info("Starting rank data preparation for datasets and models.")

            # Prepare rank data by computing the Spearman rank correlation for datasets
            datasets = DatasetRepository.get_all_datasets_with_meta_features()
            
            datasets_hpo_performances = EvaluationMaterialisedView.get_evaluations_for_all_dataset()
            logger.debug(f"Retrieved {len(datasets)} datasets and {len(datasets_hpo_performances)} dataset HPO performances.")

            performances = defaultdict(list)
            for dataset_performance in datasets_hpo_performances:
                performances[dataset_performance.dataset_idx].append(dataset_performance.avg_accuracy)

            logger.debug(f"Dataset performances aggregated: {len(performances)} dataset entries.")

            for one_dataset in datasets:
                one_dataset_idx = one_dataset.dataset_idx
                one_dataset_meta_features = one_dataset.meta_features

                for another_dataset in datasets:
                    another_dataset_idx = another_dataset.dataset_idx
                    another_dataset_meta_features = another_dataset.meta_features

                    # Skip if similarity already exists
                    if SimilarityRepository.exists_similarity(object_type="dataset", object_1_idx=one_dataset_idx, object_2_idx=another_dataset_idx):
                        logger.debug(f"Skipping existing similarity: {one_dataset_idx} <-> {another_dataset_idx}")
                        continue

                    # Retrieve performances and calculate rank score
                    one_dataset_performance = performances[one_dataset_idx]
                    another_dataset_performance = performances[another_dataset_idx]
                    rank_score = self.kendall_tau_rank_correlation(one_dataset_performance, another_dataset_performance)
                    if np.isnan(rank_score):
                        logger.warning("spearman rank 0 detected, indicating a constant performance regardless of hyperparameters, indicating problematic data entry! Discarded")
                        continue
        
                    print(f"Calculated Spearman rank score: {rank_score:.4f} for datasets {one_dataset_idx} and {another_dataset_idx}.") 
                    logger.info(f"Calculated Spearman rank score: {rank_score:.4f} for datasets {one_dataset_idx} and {another_dataset_idx}.")                 
                    # Create similarity entries (bidirectional)

                    SimilarityRepository.create_similarity(
                        object_type="dataset",
                        object_1_idx=one_dataset_idx, object_1_feature=one_dataset_meta_features, 
                        object_2_idx=another_dataset_idx, object_2_feature=another_dataset_meta_features, 
                        similarity=rank_score
                    )
                
            # Prepare rank data by computing the Spearman rank correlation for models
            models = ModelRepository.get_all_models_with_feature_vector_only()
            models_hpo_performances = EvaluationMaterialisedView.get_evaluations_for_all_models()

            logger.debug(f"Retrieved {len(models)} models and {len(models_hpo_performances)} model HPO performances.")

            performances = defaultdict(list)
            for model_performance in models_hpo_performances:
                performances[model_performance.model_idx].append(model_performance.avg_accuracy)
            logger.debug(f"Model performances aggregated: {len(performances)} model entries.")

            for one_model in models:
                one_model_idx = one_model.model_idx
                one_model_feature_vector = one_model.feature_vector

                for another_model in models:
                    another_model_idx = another_model.model_idx
                    another_model_feature_vector = another_model.feature_vector

                    # Skip if similarity already exists
                    if SimilarityRepository.exists_similarity(object_type="model", object_1_idx=one_model_idx, object_2_idx=another_model_idx):
                        logger.debug(f"Skipping existing similarity: {one_model_idx} <-> {another_model_idx}")
                        continue

                    # Retrieve performances and calculate rank score
                    one_model_performance = performances[one_model_idx]
                    another_model_performance = performances[another_model_idx]

                    rank_score = self.kendall_tau_rank_correlation(one_model_performance, another_model_performance)

                    if rank_score == float("nan"):
                        logger.warning("spearman rank 0 detected, indicating a constant performance regardless of hyperparameters, indicating problematic data entry! Discarded")
                        continue
                    logger.info(f"Calculated ktrc rank score: {rank_score:.4f} for models {one_model_idx} and {another_model_idx}.")

                    SimilarityRepository.create_similarity(
                        object_type="model",
                        object_1_idx=one_model_idx, object_1_feature=one_model_feature_vector, 
                        object_2_idx=another_model_idx, object_2_feature=another_model_feature_vector, 
                        similarity=rank_score
                    )

            logger.info("Rank data preparation completed successfully!")

        except Exception as e:
            logger.error(f"Failed to prepare rank data due to an error: {e}", exc_info=True)


    def training_for_dataset_similarity(self):
        model_path = os.path.join(os.getenv("DATASET_RANK_PREDICTION_MODEL_PATH", "./"), "dataset_similarity.pkl")

        model = DatasetSimilarityModel()

        # Load existing model if it exists
        if os.path.exists(model_path):
            model.load_model(model_path)
        else:
            logger.info("No existing model found. Training a new one.")

        # Fetch new data created after the last trained timestamp
        new_data = SimilarityRepository.get_results_after_time(object_type="dataset", created_after=datetime.min)

        if not new_data:
            logger.info(" No new dataset similarities found for training. Skipping training.")
            return

        X, y = [], []

        # Prepare features and labels
        for record in new_data:
            if record.object_1_feature and record.object_2_feature and record.similarity is not None:
                X.append(record.object_1_feature + record.object_2_feature)  # Concatenate meta-features
                y.append(record.similarity)

        if not X:
            logger.warning("No valid data for training. Exiting.")
            return

        X = np.array(X)
        y = np.array(y)

        # Train model and evaluate
        try:
            score = model.train(X, y, test_size=0.2)  
            logger.info(f"Training completed. Model Score (R²): {score:.4f}")
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return

        # Save model and metadata
        os.makedirs(os.path.dirname(model_path), exist_ok=True)  #
        model.save_model(model_path)

    def training_for_model_similarity(self):
        """Train the Random Forest model for model rank prediction."""
        
        model_path = os.path.join(os.getenv("MODEL_RANK_PREDICTION_MODEL_PATH", "./"), "model_similarity.pkl")
        model = ModelSimilarityModel()
        # Load existing model if it exists
        if os.path.exists(model_path):
            model.load_model(model_path)
        else:
            logger.info("No existing model found. Training a new one.")

        # Fetch new data since the last training
        new_data = SimilarityRepository.get_results_after_time(object_type="model", created_after=datetime.min)

        if not new_data:
            logger.info("No new dataset similarity found for training. Skipping training.")
            return

        # Prepare training data
        X, y = [], []
        for record in new_data:
            if record.object_1_feature and record.object_2_feature and record.similarity is not None:
                X.append(record.object_1_feature + record.object_2_feature)
                y.append(record.similarity)

        if not X:
            logger.warning("No valid data for training. Exiting.")
            return

        X = np.array(X)
        y = np.array(y)

        # Train model and evaluate
        try:
            score = model.train(X, y, test_size=0.2)  
            logger.info(f"Training completed. Model Score (R²): {score:.4f}")
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return

        # Save model and metadata
        os.makedirs(os.path.dirname(model_path), exist_ok=True)  #
        model.save_model(model_path)


    def kendall_tau_rank_correlation(self, d1_scores, d2_scores):

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
        
    def _spearman_rank_correlation(self, d1_scores: list | np.ndarray, d2_scores: list | np.ndarray) -> float:
        """
        Compute Spearman's rank correlation coefficient between two sets of scores.
        """
    
        # Convert to 1D numpy arrays
        d1_scores = np.ravel(np.array(d1_scores))
        d2_scores = np.ravel(np.array(d2_scores))

        # Ensure the lengths match
        if len(d1_scores) != len(d2_scores):
            raise ValueError("The two input lists must have the same length.")
        # Check if either vector is constant
        if np.std(d1_scores) == 0 and np.std(d2_scores) == 0:
            return float('nan')  # Both constant: no meaningful correlation
        # Compute Spearman's rank correlation
        correlation, _ = spearmanr(d1_scores, d2_scores)

        return float(correlation)


    def _compute_cosine_similarity(self, d1_scores: list | np.ndarray, d2_scores: list | np.ndarray) -> float:
        """
        Compute cosine similarity between two sets of scores.
        Cosine similarity measures the cosine of the angle between two vectors, 
        ranging from -1 (opposite) to 1 (identical).

        Args:
            d1_scores (list | np.ndarray): First set of scores.
            d2_scores (list | np.ndarray): Second set of scores.

        Returns:
            float: Cosine similarity between the two vectors, ranging from -1 to 1.
        """

        # Convert inputs to numpy arrays and reshape for compatibility
        d1_scores = np.array(d1_scores).reshape(1, -1)
        d2_scores = np.array(d2_scores).reshape(1, -1)
        if np.std(d1_scores) == 0 or np.std(d2_scores) == 0:
            return float("nan")  # Both constant: no meaningful correlation

        # Compute cosine similarity
        similarity = cosine_similarity(d1_scores, d2_scores)[0][0]

        return similarity