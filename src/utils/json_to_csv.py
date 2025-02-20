import json
import csv

with open("dataset.json", "r") as json_file:
    datasets = json.load(json_file)

with open("dataset.csv", "w", newline="") as csv_file:
    fieldnames = ["dataset_idx", "code", "meta_features"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for dataset in datasets:
        writer.writerow({
            "dataset_idx": dataset["dataset_num"],
            "code": json.dumps(dataset["code"]),
            "meta_features": ",".join(map(str, dataset["meta_features"]))
        })

print(" JSON to CSV conversion completed: datasets.csv")