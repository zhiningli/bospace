import json
from src.database.object import Dataset
from src.database.crud import DatasetRepository 

# Path to your exported JSON file
json_file_path = "datasets.json" 


def import_json_to_postgres():
    """Import JSON data into PostgreSQL."""
    with open(json_file_path, "r") as file:
        data = json.load(file)

    for entry in data:
        dataset_idx = entry["dataset_num"]
        code = entry["code"]
        input_size = entry["input_size"]
        num_classes = entry["num_classes"]
        meta_features = entry["meta_features"]

        result = Dataset(
            dataset_idx=dataset_idx,
            code=code,
            input_size=input_size,
            num_classes=num_classes,
            meta_features=meta_features
        )

        # ✅ Use the static method correctly
        inserted_dataset = DatasetRepository.create_dataset(
            dataset_idx=result.dataset_idx,
            code=result.code,
            input_size=result.input_size,
            num_classes=result.num_classes,
            meta_features=result.meta_features
        )

        if inserted_dataset:
            print(f"✅ Successfully inserted dataset {inserted_dataset.dataset_idx}")
        else:
            print(f"⚠️ Skipped duplicate or failed insertion for dataset {dataset_idx}")

    print("✅ JSON data imported into PostgreSQL successfully!")


    print("✅ JSON data imported into PostgreSQL successfully!")

if __name__ == "__main__":
    import_json_to_postgres()