import numpy as np

sgd_search_space = {
    "learning_rate": np.logspace(-5, -1, num=50).tolist(),
    "momentum": [0.01 * x for x in range(100)],
    "weight_decay": [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    "num_epochs": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90],
}