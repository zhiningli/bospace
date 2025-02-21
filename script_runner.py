import json
from src.database.connection import get_connection
from src.database.object.script import Script
from src.database.crud import ScriptRepository

# Path to your exported JSON file
json_file_path = "scripts.json"

def extract_sgd_config(entry):
    """Extract the best performing configuration from JSON."""
    highest_config = entry.get("highest_score_hyperparameters", {})
    highest_score = entry.get("highest_score")
    return {
        "learning_rate": highest_config.get("learning_rate", 0.0),
        "momentum": highest_config.get("momentum", 0.0),
        "weight_decay": highest_config.get("weight_decay", 0.0),
        "num_epochs": highest_config.get("num_epoches", 0),
        "highest_score": highest_score,
    }

def import_json_to_postgres():
    """Import JSON data into PostgreSQL."""
    with open(json_file_path, "r") as file:
        data = json.load(file)

    with get_connection() as conn:
        with conn.cursor() as cursor:
            for entry in data:
                try:
                    script = Script(
                        model_idx=entry["model_idx"],
                        dataset_idx= entry["dataset_idx"],
                        script_code=entry["script"],
                        sgd_best_performing_configuration=extract_sgd_config(entry)
                    )
                # ✅ 2. Use DatasetRepository to insert into DB
                    inserted_script = ScriptRepository.create_script(
                        model_idx=script.model_idx,
                        dataset_idx= script.dataset_idx,
                        script_code= script.script_code,
                        sgd_config= script.sgd_best_performing_configuration
                    )
                    if inserted_script:
                                print(f"✅ Successfully inserted models {inserted_script.script_idx}")
                    else:
                        print(f"⚠️ Skipped dataset {script.script_idx} (already exists)")
                except (ValueError, json.JSONDecodeError) as e:
                    print(f"❌ Error processing row {entry}: {e}")

            print("✅ JSON data imported into PostgreSQL successfully!")

if __name__ == "__main__":
    import_json_to_postgres()