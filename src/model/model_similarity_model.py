import numpy as np
import xgboost as xgb
import os
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

class ModelSimilarityModel:
    """XGBoost model for measuring model similarity based on feature vectors."""

    def __init__(
        self,
        hyperparameters: dict | None = None,
        random_state: int = 42
    ):
        """Initialize the ModelSimilarityModel with XGBoost hyperparameters."""
        self.model = None  # Ensure it's None before training
        self.random_state = random_state

        # Default hyperparameters
        default_params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": 6,
            "learning_rate": 0.1,
            "num_boost_round": 100,
            "seed": self.random_state
        }

        # Merge defaults with user-defined hyperparameters
        self.hyperparameters = {**default_params, **(hyperparameters or {})}


    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
        """Train the XGBoost model and evaluate its performance."""
        # Split dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        # Convert data into DMatrix format
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)


        self.model = xgb.train(
            params=self.hyperparameters,
            dtrain=dtrain,
            num_boost_round=self.hyperparameters["num_boost_round"],
            evals=[(dtrain, "train"), (dtest, "eval")],
            early_stopping_rounds=10,
            verbose_eval=False,
        )
        logger.info(f"XGBoost model trained successfully on {len(X_train)} samples.")

        # Evaluate model performance
        predictions = self.model.predict(dtest)
        rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
        print(f"Training completed. RMSE: {rmse:.4f}, y_test std: {np.std(y_test):.4f}")

        return rmse

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the trained XGBoost model."""
        if self.model is None:
            raise ValueError("Model is not trained. Please call `train` first.")
        
        dmatrix = xgb.DMatrix(X)
        return self.model.predict(dmatrix)


    def save_model(self, model_path: str, model_file: str = "xgboost_model.json"):
        """Save the trained model to a specified path."""
        if self.model is None:
            raise ValueError("No trained model to save.")
        
        os.makedirs(model_path, exist_ok=True)
        model_full_path = os.path.join(model_path, model_file)
        self.model.save_model(model_full_path)
        logger.info(f"Model saved successfully at: {model_full_path}")


    def load_model(self, model_path: str):
        """Load a trained model from a file."""
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        logger.info(f"Model loaded successfully from: {model_path}")


    def get_params(self):
        """Return current model hyperparameters."""
        return self.hyperparameters

    def set_params(self, **kwargs):
        """Update hyperparameters and reset the model."""
        self.hyperparameters.update(kwargs)
        self.model = None  # Reset model to apply new parameters
