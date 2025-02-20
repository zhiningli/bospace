import json
import csv

with open("models.json", "r") as json_file:
    models = json.load(json_file)

with open("models.csv", "w", newline="") as csv_file:
    fieldnames = ["model_idx", "code", "feature_vectors"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for model in models:
        writer.writerow({
            "model_idx": models["model_idx"],
            "code": json.dumps(models["code"]),
            "feature_vectors": ""
        })

print(" JSON to CSV conversion completed: datasets.csv")