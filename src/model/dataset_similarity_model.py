import numpy as np
from sklearn.ensemble import RandomForestRegressor

class DatasetSimilarityModel:
    """Random Forest Regressor for measuring dataset similarity 
    based on meta features from encoder to predict spearman coefficient."""

    def __init__(self, n_estimators = 200, max_depth = 20, min_samples_split = 2, min_samples_leaf = 1, random_state: int = 42):
        self._model = None
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the Random Forest model."""
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray):
        """Predict using the trained Random Forest model."""
        return self.model.predict(X)

    def get_params(self):
        """Retrieve model hyperparameters."""
        return self.model.get_params()

    def set_params(self, **params):
        """Update hyperparameters and re-instantiate the model."""
        self.hyperparameters.update(params)
        self._model = None  # Reset model to apply new parameters

