import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer

class DatasetMetaFeatureExtractor:
    def __init__(self, dataset: np.ndarray = None, labels: np.ndarray = None):
        """
        Initializes the meta-feature extractor with optional dataset and labels.

        Parameters:
            dataset (np.ndarray): Feature matrix of shape (n_samples, n_features).
            labels (np.ndarray): Target labels of shape (n_samples,).
        """
        self.dataset = dataset
        self.labels = labels

    def set_data(self, dataset: np.ndarray, labels: np.ndarray):
        """
        Sets the dataset and labels for the extractor.

        Parameters:
            dataset (np.ndarray): Feature matrix of shape (n_samples, n_features).
            labels (np.ndarray): Target labels of shape (n_samples,).
        """
        self.dataset = dataset
        self.labels = labels

    def extract_meta_features(self) -> np.ndarray:
        """
        Extracts 46 meta-features from the dataset and labels.

        Returns:
            np.ndarray: An array containing all the extracted meta-features.
        """
        if self.dataset is None or self.labels is None:
            raise ValueError("Dataset and labels must be set before extracting meta-features.")

        meta_features = []

        # Simple meta-features 1 - 23
        meta_features.extend(self._num_samples_metafeatures())
        meta_features.extend(self._num_features_metafeatures())
        meta_features.append(self._num_classes())

        meta_features.extend(self._num_samples_with_missing_values_metafeatures())
        meta_features.extend(self._num_features_with_missing_values_metafeatures())
        meta_features.extend(self._num_missing_values_metafeatures())
        
        meta_features.extend(self._num_categorical_features_metafeatures())
        meta_features.extend(self._dataset_dimensionality_metafeaures())

        meta_features.extend(self._compute_class_probabilities())

        # Information theoretic feature 24
        meta_features.append(self._compute_class_entropy())

        # Statistical features 25- 37
        meta_features.extend(self._categorical_value_states())
        meta_features.extend(self._kurtosis_stats())
        meta_features.extend(self._skew_stats())
        
        # PCA metafeatures 38 - 40
        meta_features.extend(self._pca_stats())

        # landmark metafeatures 41 - 46
        meta_features.extend(self._landmarking_meta_features())
        return np.array(meta_features)

    # === Simple Meta-Features ===
    def _num_samples_metafeatures(self) -> list:
        return [self.dataset.shape[0], np.log1p(self.dataset.shape[0])]

    def _num_features_metafeatures(self) -> list:
        return [self.dataset.shape[1], np.log1p(self.dataset.shape[1])]

    def _num_classes(self) -> int:
        return len(np.unique(self.labels))

    def _num_samples_with_missing_values_metafeatures(self) -> list:
        return [np.sum(np.isnan(self.dataset).any(axis=1)), 
                np.sum(np.isnan(self.dataset).any(axis=1)) / self.dataset.shape[0]]
    
    def _num_features_with_missing_values_metafeatures(self) -> list:
        return [np.sum(np.isnan(self.dataset).any(axis=0)), 
            np.sum(np.isnan(self.dataset).any(axis=0)) / self.dataset.shape[1]]
    
    def _num_missing_values_metafeatures(self) -> list:
        return [np.sum(np.isnan(self.dataset)), np.sum(np.isnan(self.dataset)) / (self.dataset.shape[0] * self.dataset.shape[1])]

    def _num_categorical_features_metafeatures(self, threshold: int = 10) -> list:
        total_features = self.dataset.shape[1]
        num_categorical = 0
        for feature in range(total_features):
            unique_values = len(np.unique(self.dataset[:, feature]))
            if unique_values <= threshold:
                num_categorical += 1
        
        num_numerical = total_features - num_categorical
        
        return [num_categorical, num_numerical, num_categorical / total_features, num_numerical/ total_features]
    

    def _dataset_dimensionality_metafeaures(self) -> list:
        return [self.dataset.shape[1] / self.dataset.shape[0], np.log1p(self.dataset.shape[1] / self.dataset.shape[0]), self.dataset.shape[0] / self.dataset.shape[1], np.log1p(self.dataset.shape[0] / self.dataset.shape[1])]

    def _compute_class_probabilities(self) -> list:
        _, class_count = np.unique(self.labels, return_counts=True)
        class_probs = class_count / len(self.labels)
        return [np.min(class_probs), np.max(class_probs), np.mean(class_probs), np.std(class_probs)]

    def _compute_class_entropy(self) -> float:
        """
        Computes the class entropy of the dataset labels.
        """
        _, class_counts = np.unique(self.labels, return_counts=True)
        class_probs = class_counts / len(self.labels)
        return -np.sum(class_probs * np.log2(class_probs))


    def _categorical_value_states(self, threshold=10) -> list:
        categorical_values = []

        for feature in range(self.dataset.shape[1]):
            unique_values = len(np.unique(self.dataset[:, feature]))
            if unique_values <= threshold:
                categorical_values.append(unique_values)
        
        return [np.min(categorical_values) if categorical_values else 0, np.max(categorical_values) if categorical_values else 0, np.mean(categorical_values) if categorical_values else 0, np.std(categorical_values) if categorical_values else 0, np.sum(categorical_values) if categorical_values else 0]
    

    def _kurtosis_stats(self) -> list:
        kurtosis_values = kurtosis(self.dataset, axis=0, nan_policy="omit")

        return [np.nanmin(kurtosis_values), np.nanmax(kurtosis_values), np.nanmean(kurtosis_values), np.nanstd(kurtosis_values)]
    

    def _skew_stats(self) -> list:
        skewness_values = skew(self.dataset, axis=0, nan_policy="omit")

        return [np.nanmin(skewness_values), np.nanmax(skewness_values), np.nanmean(skewness_values), np.nanstd(skewness_values)]
    

    def _pca_stats(self) -> list:
        imputer = SimpleImputer(strategy="mean")  # Other options: "median", "most_frequent"
        X_imputed = imputer.fit_transform(self.dataset)

        pca = PCA()
        pca.fit(X_imputed)

        pca_95 = np.sum(np.cumsum(pca.explained_variance_ratio_) <= 0.95)

        first_pc = pca.transform(X_imputed)[:, 0]

        pca_skewness_first_pc = skew(first_pc)
        pca_kurtosis_first_pc = kurtosis(first_pc)

        return [pca_95, pca_skewness_first_pc, pca_kurtosis_first_pc]
    

    def _landmarking_meta_features(self) -> list:
        # === Handle Missing Values (Mean Imputation) ===
        imputer = SimpleImputer(strategy="mean")  # Fill missing values with column mean
        X_imputed = imputer.fit_transform(self.dataset)  # Ensure dataset is clean

        meta_features = []

        # One Nearest Neighbor
        knn = KNeighborsClassifier(n_neighbors=1)
        meta_features.append(np.mean(cross_val_score(knn, X_imputed, self.labels, cv=5)))

        # Linear Discriminant Analysis
        lda = LinearDiscriminantAnalysis()
        meta_features.append(np.mean(cross_val_score(lda, X_imputed, self.labels, cv=5)))

        # Naive Bayes
        nb = GaussianNB()
        meta_features.append(np.mean(cross_val_score(nb, X_imputed, self.labels, cv=5)))

        # Decision Tree
        dt = DecisionTreeClassifier()
        meta_features.append(np.mean(cross_val_score(dt, X_imputed, self.labels, cv=5)))

        # Decision Stump (Decision Tree with depth = 1)
        decision_node = DecisionTreeClassifier(max_depth=1)
        meta_features.append(np.mean(cross_val_score(decision_node, X_imputed, self.labels, cv=5)))

        # Random Node Learner (random splits)
        random_node = DecisionTreeClassifier(splitter='random')
        meta_features.append(np.mean(cross_val_score(random_node, X_imputed, self.labels, cv=5)))

        return meta_features