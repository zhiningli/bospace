import numpy as np
def convert_bo_output_to_json_format(constrained_search_space, accuracies, best_y, best_candidates):
    
    # ✅ Ensure accuracies are float
    accuracies = list(map(float, accuracies))
    
    # ✅ Ensure best_candidates is a flattened list
    if isinstance(best_candidates, np.ndarray):
        best_candidates = best_candidates.flatten().tolist()
    elif isinstance(best_candidates, list):
        best_candidates = [int(x) for x in best_candidates]
    
    print(f"Processed best_candidates: {best_candidates}")

    # ✅ Validate indexing
    try:
        best_sgd_configurations = {
            "learning_rate": constrained_search_space["learning_rate"][best_candidates[0]],
            "momentum": constrained_search_space["momentum"][best_candidates[1]],
            "weight_decay": constrained_search_space["weight_decay"][best_candidates[2]],
            "num_epochs": constrained_search_space["num_epochs"][best_candidates[3]],
            "highest_score": float(best_y)
        }
    except (IndexError, KeyError, TypeError) as e:
        raise ValueError(f"❌ Error accessing constrained_search_space: {e}")

    return {
        "best_configuration": best_sgd_configurations,
        "optimisor": "SGD",
        "accuracies": accuracies
    }
