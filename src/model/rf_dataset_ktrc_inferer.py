import numpy as np
from sklearn.ensemble import RandomForestRegressor


class RF_dataset_ktrc_inferer:
    r"""This rf-regressor measures datasets similariy by estimating two dataset' kendall tau rank coefficient
        based on meta features extracted by dataset embedder"""

    def __init__(self, random_state = 42):
        self._model = None
        self.random_state = random_state
    
    @property
    def model(self):
        """Lazy loading using getter function"""
        if self._model is None:
            self._model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=2,
            random_state=self.random_state
        )
        return self._model

    def train(self, X_train: np.ndarray, y_train: np.ndarray ) -> None:
        self._model.fit(X_train, y_train)

    def predict(self, X:np.ndarray):
        return self._model.predict(X)
    
