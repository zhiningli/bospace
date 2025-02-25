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
                    rank_score = self._spearman_rank_correlation(one_dataset_performance, another_dataset_performance)
                    if np.isnan(rank_score):
                        logger.warning("spearman rank 0 detected, indicating a constant performance regardless of hyperparameters, indicating problematic data entry! Discarded")
                        continue
        
                    print(f"Calculated Spearman rank score: {rank_score:.4f} for datasets {one_dataset_idx} and {another_dataset_idx}.") 
                    logger.info(f"Calculated Spearman rank score: {rank_score:.4f} for datasets {one_dataset_idx} and {another_dataset_idx}.")                 
                    # Create similarity entries (bidirectional)
                    for obj1, obj2, idx1, idx2 in [
                        (one_dataset_meta_features, another_dataset_meta_features, one_dataset_idx, another_dataset_idx),
                        (another_dataset_meta_features, one_dataset_meta_features, another_dataset_idx, one_dataset_idx)
                    ]:
                        SimilarityRepository.create_similarity(
                            object_type="dataset",
                            object_1_idx=idx1, object_1_feature=obj1, 
                            object_2_idx=idx2, object_2_feature=obj2, 
                            similarity=rank_score
                        )
                        logger.debug(f"Created similarity for datasets: {idx1} <-> {idx2}")
                
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

                    rank_score = self._spearman_rank_correlation(one_model_performance, another_model_performance)
                    if rank_score == float("nan"):
                        logger.warning("spearman rank 0 detected, indicating a constant performance regardless of hyperparameters, indicating problematic data entry! Discarded")
                        continue
                    logger.info(f"Calculated Spearman rank score: {rank_score:.4f} for models {one_model_idx} and {another_model_idx}.")
                    # Create similarity entries (bidirectional)
                    for obj1, obj2, idx1, idx2 in [
                        (one_model_feature_vector, another_model_feature_vector, one_model_idx, another_model_idx),
                        (another_model_feature_vector, one_model_feature_vector, another_model_idx, one_model_idx)
                    ]:
                        SimilarityRepository.create_similarity(
                            object_type="model",
                            object_1_idx=idx1, object_1_feature=obj1, 
                            object_2_idx=idx2, object_2_feature=obj2, 
                            similarity=rank_score
                        )
                        logger.debug(f"Created similarity for models: {idx1} <-> {idx2}")

            logger.info("Rank data preparation completed successfully!")

        except Exception as e:
            logger.error(f"Failed to prepare rank data due to an error: {e}", exc_info=True)


    def training_RFRegressor_for_dataset_rank_prediction(self):
        model_path = os.getenv("DATASET_RANK_PREDICTION_MODEL_PATH", "./")
        model_file = os.path.join(model_path, "dataset_similarity_model.joblib")
        metadata_file = os.path.join(model_path, "dataset_similarity_model_metadata.json")

        # Load metadata if available
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, "r") as meta_file:
                    metadata = json.load(meta_file)
                    last_trained_at = datetime.fromisoformat(metadata.get("last_trained_at", datetime.min.isoformat()))
                logger.info(f"Metadata loaded. Last trained at: {last_trained_at}")
            except (json.JSONDecodeError, KeyError, ValueError):
                logger.warning(" Failed to parse metadata. Using defaults.")
                last_trained_at = datetime.min
        else:
            last_trained_at = datetime.min

        # Load existing model or create a new one
        if os.path.exists(model_file):
            model = joblib.load(model_file)
            logger.info("Loaded existing RandomForest model.")
        else:
            model = DatasetSimilarityModel()
            logger.warning("No existing model found. Created a new RandomForest model.")

        # Fetch new data created after the last trained timestamp
        new_data = SimilarityRepository.get_results_after_time(object_type="dataset", created_after=last_trained_at)

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

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model.fit(X_train, y_train)
        print((f"Model trained successfully on {len(X_train)} samples."))
        logger.info(f"Model trained successfully on {len(X_train)} samples.")

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        logger.info(f"Model evaluation complete. Mean Squared Error: {mse:.4f}")
        print(f"Shape of y_test: {y_test.shape}, Shape of y_pred: {y_pred.shape}")

        print(f"Model evaluation complete. Mean Squared Error: {mse:.4f}, std of label is {np.std(y_pred)}")
        os.makedirs(model_path, exist_ok=True)
        joblib.dump(model, model_file)
        logger.info(f"Model saved to {model_file}")

        # Update metadata with current training details
        metadata = {
            "last_trained_at": datetime.now().isoformat(),
            "model_type": "RandomForestRegressor",
            "hyperparameters": model.get_params(),
            "performance": {"mse": mse}
        }

        # Write metadata file
        with open(metadata_file, "w") as meta_file:
            json.dump(metadata, meta_file, indent=4)
        logger.info(f"Metadata updated and saved to {metadata_file}.")



    def training_XGBoost_for_model_rank_prediction(self):
        """Train the XGBoost model for model rank prediction."""
        model_path = os.getenv("MODEL_RANK_PREDICTION_MODEL_PATH", "./")
        model_file = os.path.join(model_path, "xgboost_model.json")
        metadata_file = os.path.join(model_path, "model_similarity_model_metadata.json")

        model = ModelSimilarityModel()
        last_trained_at = datetime.min

        if os.path.exists(model_file) and os.path.exists(metadata_file):
            try:
                model.load_model(model_file)
                with open(metadata_file, "r") as meta_file:
                    metadata = json.load(meta_file)
                    last_trained_at = datetime.fromisoformat(metadata.get("last_trained_at", str(datetime.min)))
                    logger.info(f"Loaded existing model trained at: {last_trained_at}.")
            except (json.JSONDecodeError, FileNotFoundError, KeyError):
                logger.warning("Failed to load existing model or metadata. Training from scratch.")
        
        # Fetch new data since the last training
        new_data = SimilarityRepository.get_results_after_time(object_type="model", created_after=last_trained_at)

        if not new_data:
            logger.info("No new dataset similarity found for training XGBoost. Skipping training.")
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
            rmse = model.train(X, y, test_size=0.2)
            
           
            logger.info(f"Training completed. RMSE: {rmse:.4f}")
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return

        # Save model and metadata
        os.makedirs(model_path, exist_ok=True)
        model.save_model(model_path)

        metadata = {
            "last_trained_at": datetime.now().isoformat(),
            "model_type": "XGBoost",
            "hyperparameters": model.get_params()
        }

        try:
            with open(metadata_file, "w") as meta_file:
                json.dump(metadata, meta_file, indent=4)
            logger.info(f"Model and metadata saved successfully. Last trained at: {metadata['last_trained_at']}.")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")


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