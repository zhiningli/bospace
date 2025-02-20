import csv
import json
from src.database.object.model import Model
from src.database.crud.model_crud import ModelRepository

def insert_csv_to_postgres(csv_path: str):
    """Insert datasets from a CSV file into PostgreSQL using Dataset dataclass."""
    with open(csv_path, "r") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            try:
                # ✅ 1. Create Dataset object from each row
                model = Model(
                    model_idx=int(row["model_idx"]),
                    code=row["code"],
                    feature_vector=row["feature_vectors"]
                )

                # ✅ 2. Use DatasetRepository to insert into DB
                inserted_model = ModelRepository.create_model(
                    model_idx=model.model_idx,
                    code=model.code,
                    feature_vector=model.feature_vector
                )

                if inserted_model:
                    print(f"✅ Successfully inserted models {model.model_idx}")
                else:
                    print(f"⚠️ Skipped dataset {model.model_idx} (already exists)")

            except (ValueError, json.JSONDecodeError) as e:
                print(f"❌ Error processing row {row}: {e}")

if __name__ == "__main__":
    csv_path = "models.csv"  # Update with the correct path
    insert_csv_to_postgres(csv_path)
