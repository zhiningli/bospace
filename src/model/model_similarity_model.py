import numpy as np
import os
import logging
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("service")

class ModelSimilarityModel:
    """Random Forest model for predicting similarity with PCA-based feature reduction."""

    def __init__(self, hyperparameters: dict | None = None, random_state: int = 42):
        """Initialize the ModelSimilarityModel with Random Forest, PCA, and scaler."""
        self.model = None
        self.pca = None  
        self.random_state = random_state

        # Default hyperparameters optimized for similarity ranking
        default_params = {
            "pca_components": 128,   # Reduce dimensionality to 64
            "n_estimators": 100,    # Number of trees in Random Forest
            "max_depth": None,      # Allow tree to grow fully
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

        # Apply PCA
        self.pca = PCA(n_components=self.hyperparameters["pca_components"])
        X_train_pca = self.pca.fit_transform(X_train)
        X_test_pca = self.pca.transform(X_test)

        # Train the Random Forest model
        print("🌲 Training Random Forest Regressor...")
        self.model.fit(X_train_pca, y_train)
        print("✅ Training complete!")

        # Compute evaluation score (optional)
        score = self.model.score(X_test_pca, y_test)

        return score

    def predict(self, X: np.ndarray) -> float:
        """Predict the similarity score for a given feature vector."""
        if self.model is None or self.pca is None:
            raise ValueError("⚠️ Model, PCA, or Scaler not initialized. Load a trained model first.")

        X = np.atleast_2d(X)

        X_pca = self.pca.transform(X)

        # Predict using RandomForestRegressor
        y_pred = self.model.predict(X_pca)

        return float(y_pred)

    def save_model(self, file_path="model_similarity.pkl"):
        """Save the trained model, PCA, and scaler."""
        joblib.dump({"model": self.model, "pca": self.pca}, file_path)
        print(f"💾 Model saved at {file_path}")

    def load_model(self, file_path="model_similarity.pkl"):
        """Load the trained model, PCA, and scaler."""
        data = joblib.load(file_path)
        self.model = data["model"]
        self.pca = data["pca"]
        print(f"📂 Model loaded from {file_path}")
