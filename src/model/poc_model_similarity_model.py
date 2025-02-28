import numpy as np
import os
import joblib  # For saving/loading model
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import kendalltau

class SVRSimilarityModel:
    """SVR model for predicting similarity based on feature vectors, optimized with Kendall Tau ranking behavior."""

    def __init__(self, C=10.0, gamma="scale", kernel="rbf", pca_components=256, random_state=42):
        """Initialize SVR model with PCA and StandardScaler."""
        self.model = SVR(kernel=kernel, C=C, gamma=gamma)  # Use RBF kernel for non-linearity
        self.pca = PCA(n_components=pca_components)
        self.scaler = StandardScaler()
        self.random_state = random_state

    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
        """Train the SVR model."""
        print(f"🚀 Starting SVR training with dataset of shape: {X.shape}, labels shape: {y.shape}")

        # ✅ Transform y to be in [0,1]
        y = (y + 1) / 2  

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        # Apply StandardScaler
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Apply PCA
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)

        # Train SVR
        self.model.fit(X_train_pca, y_train)

        # Predict on test set
        preds = self.model.predict(X_test_pca)

        # Compute Kendall Tau
        tau, _ = kendalltau(y_test, preds)
        print(f"✅ Final Eval Kendall Tau: {tau:.4f}")

        return tau

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the trained SVR model."""
        if self.model is None:
            raise ValueError("⚠️ Model is not trained. Please call `train` first.")

        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)

        return self.model.predict(X_pca)

    def save_model(self, model_path: str, model_file: str = "svr_model.pkl"):
        """Save the trained model to a specified path."""
        if self.model is None:
            raise ValueError("No trained model to save.")

        os.makedirs(model_path, exist_ok=True)
        model_full_path = os.path.join(model_path, model_file)

        # Save everything (SVR, PCA, Scaler)
        joblib.dump({"model": self.model, "pca": self.pca, "scaler": self.scaler}, model_full_path)
        print(f"✅ Model saved successfully at: {model_full_path}")

    def load_model(self, model_path: str):
        """Load a trained model from a file."""
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"⚠️ Model file not found: {model_path}")

        loaded_data = joblib.load(model_path)
        self.model = loaded_data["model"]
        self.pca = loaded_data["pca"]
        self.scaler = loaded_data["scaler"]

        print(f"✅ Model loaded successfully from: {model_path}")
