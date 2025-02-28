import numpy as np
import logging
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("service")

class DatasetSimilarityModel:
    """Random Forest model for predicting similarity with metafeatures."""

    def __init__(self, hyperparameters: dict | None = None, random_state: int = 42):
        """Initialize the ModelSimilarityModel with Random Forest"""
        self.model = None
        self.random_state = random_state

        # Default hyperparameters optimized for similarity ranking
        default_params = {
            'max_depth': None, 
            'max_features': 'sqrt', 
            'min_samples_leaf': 1, 
            'min_samples_split': 2, 
            'n_estimators': 200  
        }
        
        self.hyperparameters = {**default_params, **(hyperparameters or {})}
        
        self.model = RandomForestRegressor(
            n_estimators=self.hyperparameters["n_estimators"],
            max_depth=self.hyperparameters["max_depth"],
            random_state=self.random_state
        )

    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
        """Train the model using RandomForestRegressor."""
        print(f"🚀 Starting training with dataset of shape: {X.shape}, labels shape: {y.shape}")

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )


        # Train the Random Forest model
        print("🌲 Training Random Forest Regressor...")
        self.model.fit(X_train, y_train)
        print("✅ Training complete!")

        # Compute evaluation score (optional)
        score = self.model.score(X_test, y_test)

        return score

    def predict(self, X: np.ndarray) -> float:
        """Predict the similarity score for a given feature vector."""
        if self.model is None:
            raise ValueError("⚠️ Model, PCA, or Scaler not initialized. Load a trained model first.")

        X = np.atleast_2d(X)
        y_pred = self.model.predict(X)

        return float(y_pred)

    def save_model(self, file_path="dataset_similarity.pkl"):
        """Save the trained model."""
        joblib.dump({"model": self.model}, file_path)
        print(f"💾 Model saved at {file_path}")

    def load_model(self, file_path="dataset_similarity.pkl"):
        """Load the trained model"""
        data = joblib.load(file_path)
        self.model = data["model"]
        print(f"📂 Model loaded from {file_path}")
