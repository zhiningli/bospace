"""
This service handles preparation of rank results, training XGBoost model for model rank prediction and training RFRegressor model for dataset rank prediction
"""
from src.database.crud import DatasetRepository, SimilarityRepository, HPEvaluationRepository, ModelRepository
import numpy as np
import os
from dotenv import load_dotenv
from collections import defaultdict
import joblib
from scipy.stats import spearmanr

import logging
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import xgboost as xgb
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
            datasets = DatasetRepository.get_all_dataset()
            datasets_hpo_performances = HPEvaluationRepository.get_average_accuracy_per_JSON_array_index_group_by_dataset()

            logger.debug(f"Retrieved {len(datasets)} datasets and {len(datasets_hpo_performances)} dataset HPO performances.")

            performances = defaultdict(list)
            for dataset_performance in datasets_hpo_performances:
                performances[dataset_performance[0]].append(dataset_performance[1])

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
                    logger.info(f"Calculated Spearman rank score: {rank_score:.4f} for datasets {one_dataset_idx} and {another_dataset_idx}.")

                    # Create similarity entries (bidirectional)
                    for obj1, obj2, idx1, idx2 in [
                        (one_dataset_meta_features, another_dataset_meta_features, one_dataset_idx, another_dataset_idx),
                        (another_dataset_meta_features, one_dataset_meta_features, another_dataset_idx, one_dataset_idx)
                    ]:
                        SimilarityRepository.create_similarity(
                            object_type="dataset",
                            object_1=obj1, object_1_idx=idx1,
                            object_2=obj2, object_2_idx=idx2,
                            similarity=rank_score
                        )
                        logger.debug(f"Created similarity for datasets: {idx1} <-> {idx2}")

            # Prepare rank data by computing the Spearman rank correlation for models
            models = ModelRepository.get_all_models()
            models_hpo_performances = HPEvaluationRepository.get_average_accuracy_per_JSON_array_index_group_by_model()

            logger.debug(f"Retrieved {len(models)} models and {len(models_hpo_performances)} model HPO performances.")

            performances = defaultdict(list)
            for model_performance in models_hpo_performances:
                performances[model_performance[0]].append(model_performance[1])

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
                    logger.info(f"Calculated Spearman rank score: {rank_score:.4f} for models {one_model_idx} and {another_model_idx}.")

                    # Create similarity entries (bidirectional)
                    for obj1, obj2, idx1, idx2 in [
                        (one_model_feature_vector, another_model_feature_vector, one_model_idx, another_model_idx),
                        (another_model_feature_vector, one_model_feature_vector, another_model_idx, one_model_idx)
                    ]:
                        SimilarityRepository.create_similarity(
                            object_type="model",
                            object_1=obj1, object_1_idx=idx1,
                            object_2=obj2, object_2_idx=idx2,
                            similarity=rank_score
                        )
                        logger.debug(f"Created similarity for models: {idx1} <-> {idx2}")

            logger.info("Rank data preparation completed successfully!")

        except Exception as e:
            logger.error(f"Failed to prepare rank data due to an error: {e}", exc_info=True)


    def training_RFRegressor_for_dataset_rank_prediction(self):
        model_path = os.getenv("DATASET_RANK_PREDICTION_MODEL_PATH", "./")
        model_file = os.path.join(model_path, "best_random_forest_model.pkl")
        metadata_file = os.path.join(model_path, "random_forest_model_metadata.json")

        # Load metadata if available
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, "r") as meta_file:
                    metadata = json.load(meta_file)
                    last_trained_at = datetime.fromisoformat(metadata.get("last_trained_at", datetime.min.isoformat()))
                    hyperparameters = metadata.get("hyperparameters", {})
                logger.info(f"Metadata loaded. Last trained at: {last_trained_at}")
            except (json.JSONDecodeError, KeyError, ValueError):
                logger.warning(" Failed to parse metadata. Using defaults.")
                last_trained_at = datetime.min
                hyperparameters = {}
        else:
            last_trained_at = datetime.min
            hyperparameters = {}

        # Load existing model or create a new one
        if os.path.exists(model_file):
            model = joblib.load(model_file)
            logger.info("Loaded existing RandomForest model.")
        else:
            model = RandomForestRegressor(
                n_estimators=hyperparameters.get("n_estimators", 200),
                max_depth=hyperparameters.get("max_depth", 20),
                min_samples_split=hyperparameters.get("min_samples_split", 2),
                min_samples_leaf=hyperparameters.get("min_samples_leaf", 1),
                random_state=42
            )
            logger.warning("No existing model found. Created a new RandomForest model.")

        # Fetch new data created after the last trained timestamp
        new_data = SimilarityRepository.get_results_after_time(object_type="dataset", created_after=last_trained_at)

        if not new_data:
            logger.info(" No new dataset similarities found for training. Skipping training.")
            return

        X, y = [], []

        # Prepare features and labels
        for record in new_data:
            if record.object_1 and record.object_2 and record.similarity is not None:
                X.append(record.object_1 + record.object_2)  # Concatenate meta-features
                y.append(record.similarity)

        if not X:
            logger.warning("No valid data for training. Exiting.")
            return

        X = np.array(X)
        y = np.array(y)

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model.fit(X_train, y_train)
        logger.info(f"Model trained successfully on {len(X_train)} samples.")

        # Evaluate the model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        logger.info(f"Model evaluation complete. Mean Squared Error: {mse:.4f}")

        # Save the model and metadata
        os.makedirs(model_path, exist_ok=True)
        joblib.dump(model, model_file)
        logger.info(f"Model saved to {model_file}")

        # Update metadata with current training details
        metadata = {
            "last_trained_at": datetime.now().isoformat(),
            "model_type": "RandomForestRegressor",
            "hyperparameters": {
                "n_estimators": model.n_estimators,
                "max_depth": model.max_depth,
                "min_samples_split": model.min_samples_split,
                "min_samples_leaf": model.min_samples_leaf
            },
            "performance": {"mse": mse}
        }

        # Write metadata file
        with open(metadata_file, "w") as meta_file:
            json.dump(metadata, meta_file, indent=4)
        logger.info(f"✅ Metadata updated and saved to {metadata_file}.")



    def training_XGBoost_for_model_rank_prediction(self):
        """Training the XGBoost model for model rank prediction."""
        model_path = os.getenv("MODEL_RANK_PREDICTION_MODEL_PATH", "./")
        model_file = os.path.join(model_path, "best_XGBoost_model.json")
        metadata_file = os.path.join(model_path, "best_XGBoost_model_metadata.json")

        # Check if model and metadata exist
        if os.path.exists(model_file) and os.path.exists(metadata_file):
            # Load existing model
            model = xgb.Booster()
            model.load_model(model_file)

            # Load metadata
            try:
                with open(metadata_file, "r") as meta_file:
                    metadata = json.load(meta_file)
                    last_trained_at = datetime.fromisoformat(metadata['last_trained_at'])
                    hyperparameters = metadata['hyperparameters']
            except (json.JSONDecodeError, KeyError):
                logger.error("Failed to parse metadata file. Training from scratch.")
                last_trained_at = datetime.min
                hyperparameters = {
                    "objective": "reg:squarederror",
                    "eval_metric": "rmse",
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "n_estimators": 100,
                    "num_boost_round": 100
                }
        else:
            # Train from scratch
            logger.info("No existing model found. Training from scratch.")
            model = None
            last_trained_at = datetime.min
            hyperparameters = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 100,
                "num_boost_round": 100
            }

        # Fetch new data since the last training
        new_data = SimilarityRepository.get_results_after_time(object_type="model", created_after=last_trained_at)

        if not new_data:
            logger.info("No new dataset similarity found for training XGBoost. Skipping training.")
            return

        X, y = [], []

        # Prepare features and labels
        for record in new_data:
            if record.object_1 and record.object_2 and record.similarity is not None:
                X.append(record.object_1 + record.object_2)
                y.append(record.similarity)

        if not X:
            logger.warning("No valid data for training. Exiting.")
            return

        X = np.array(X)
        y = np.array(y)

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        model = xgb.train(hyperparameters, dtrain, num_boost_round=hyperparameters['num_boost_round'])

        logger.info(f"✅ XGBoost model successfully trained on {len(X_train)} samples.")

        # Evaluate model performance
        predictions = model.predict(dtest)
        rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
        logger.info(f"📊 Model RMSE on test set: {rmse:.4f}")

        # Save the model and metadata
        os.makedirs(model_path, exist_ok=True)
        model.save_model(model_file)

        metadata = {
            "last_trained_at": datetime.now().isoformat(),
            "model_type": "XGBoost",
            "hyperparameters": hyperparameters,
            "num_boost_round": hyperparameters['num_boost_round']
        }

        # Write metadata file
        with open(metadata_file, "w") as meta_file:
            json.dump(metadata, meta_file, indent=4)

        logger.info(f"✅ Model and metadata saved. Last trained at: {metadata['last_trained_at']}.")

            


    def _spearman_rank_correlation(self, d1_scores: list | np.ndarray, d2_scores: list | np.ndarray) -> float:
        """
        Compute Spearman's rank correlation coefficient between two sets of scores.
        Spearman's correlation is invariant to the input order and measures 
        the strength and direction of the monotonic relationship between two variables.

        Args:
            d1_scores (list | np.ndarray): First set of scores.
            d2_scores (list | np.ndarray): Second set of scores.

        Returns:
            float: Spearman's rank correlation coefficient, ranging from -1 to 1.
        """

        # Convert inputs to numpy arrays
        d1_scores = np.array(d1_scores)
        d2_scores = np.array(d2_scores)

        # Ensure the lengths match
        if len(d1_scores) != len(d2_scores):
            raise ValueError("The two input lists must have the same length.")

        # Compute Spearman's rank correlation
        correlation, _ = spearmanr(d1_scores, d2_scores)

        return correlation
