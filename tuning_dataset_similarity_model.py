from src.database import SimilarityRepository
from datetime import datetime
import numpy as np
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split


training_data = SimilarityRepository.get_results_after_time(object_type="dataset", created_after=datetime.min)

# Prepare dataset
X, y = [], []
for record in training_data:
    if record.object_1_feature and record.object_2_feature and record.similarity is not None:
        X.append(record.object_1_feature + record.object_2_feature)  # Concatenating features
        y.append(record.similarity)

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Split dataset into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Define hyperparameter grid
param_grid = {
    "n_estimators": [50, 100, 200],   # Number of trees
    "max_depth": [None, 10, 20, 30],  # Maximum depth of each tree
    "min_samples_split": [2, 5, 10],  # Minimum samples to split an internal node
    "min_samples_leaf": [1, 2, 4],    # Minimum samples per leaf node
    "max_features": ["sqrt", "log2"], # Number of features to consider for splits
}

# Initialize the model
rf = RandomForestRegressor(random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    scoring="neg_mean_squared_error",  # Minimize MSE
    n_jobs=-1,  # Use all available cores
    verbose=2   # Print progress
)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Print best hyperparameters
print(f"Best hyperparameters: {grid_search.best_params_}")

# Get best model
best_rf = grid_search.best_estimator_

# Evaluate on test set
test_score = best_rf.score(X_test, y_test)
print(f"Test R^2 Score: {test_score:.4f}")







