import json

# Load the JSON data
with open("evaluation_results_bo.json", "r") as f:
    data = json.load(f)

counter = 0

for key, results in data.items():
    uncon = results["unconstrained_results"]
    cons = results["constrained_results"]

    # Compute threshold (target low accuracy)
    uncon_threshold = 0.95 * uncon["best_y"]
    cons_threshold = 0.95 * cons["best_y"]

    # Find index where accuracy >= threshold
    def find_first_index(accuracies, threshold):
        for i, acc in enumerate(accuracies):
            if acc >= threshold:
                return i
        return None

    uncon_index = find_first_index(uncon["accuracies"], uncon_threshold)
    cons_index = find_first_index(cons["accuracies"], cons_threshold)

    if (
        uncon_index is not None and
        cons_index is not None and
        uncon_index > cons_index
    ):
        counter += 1

print(f"Counter: {counter}")
