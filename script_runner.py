import csv
import json
from src.database.object.dataset import Dataset
from src.database.crud.dataset_crud import DatasetRepository

def insert_csv_to_postgres(csv_path: str):
    """Insert datasets from a CSV file into PostgreSQL using Dataset dataclass."""
    with open(csv_path, "r") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            try:
                # ✅ 1. Create Dataset object from each row
                dataset = Dataset(
                    dataset_idx=int(row["dataset_idx"]),
                    code=row["code"],
                    meta_features=[float(x) for x in row["meta_features"].split(",")]
                )


                # ✅ 2. Use DatasetRepository to insert into DB
                inserted_dataset = DatasetRepository.create_dataset(
                    dataset_idx=dataset.dataset_idx,
                    code=dataset.code,
                    meta_features=dataset.meta_features
                )

                if inserted_dataset:
                    print(f"✅ Successfully inserted dataset {inserted_dataset.dataset_idx}")
                else:
                    print(f"⚠️ Skipped dataset {dataset.dataset_idx} (already exists)")

            except (ValueError, json.JSONDecodeError) as e:
                print(f"❌ Error processing row {row}: {e}")

if __name__ == "__main__":
    csv_path = "dataset.csv"  # Update with the correct path
    insert_csv_to_postgres(csv_path)
