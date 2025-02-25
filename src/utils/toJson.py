def convert_bo_output_to_json_format(constrained_search_space, accuracies, best_y, best_candidates):
    accuracies = list(map(float, accuracies))
    best_candidates = list(map(int, best_candidates.flatten().tolist()))
    best_sgd_configurations = {
        "learning_rate": constrained_search_space["learning_rate"][int(best_candidates[0])],
        "momentum": constrained_search_space["momentum"][int(best_candidates[1])],
        "weight_decay": constrained_search_space["weight_decay"][int(best_candidates[2])],
        "num_epochs": constrained_search_space["num_epochs"][int(best_candidates[3])],
        "highest_score": float(best_y)
    }

    return {
        "best_configuration": best_sgd_configurations,
        "optimisor": "SGD",
        "accuracies": accuracies
    }