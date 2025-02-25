import json
from src.database.object import HP_evaluation
from src.database.crud import HPEvaluationRepository

# Path to your exported JSON file
json_file_path = "new_dataset_hp_evaluations.json"

def import_json_to_postgres():
    """Import JSON data into PostgreSQL."""
    with open(json_file_path, "r") as file:
        data = json.load(file)

    for entry in data:
        dataset_idx = entry["dataset_idx"]
        model_idx = entry["model_idx"]
        results = entry["result"]

        hp_evaluation = HP_evaluation(
            model_idx= model_idx,
            dataset_idx= dataset_idx,
            results = results
        )
    # ✅ 2. Use DatasetRepository to insert into DB
        inserted_result = HPEvaluationRepository.create_hp_evaluation(
            model_idx = hp_evaluation.model_idx,
            dataset_idx = hp_evaluation.dataset_idx,
            results= hp_evaluation.results,
        )

        if inserted_result:
            print(f"✅ Successfully inserted hp_result {inserted_result.hp_evaluation_id}")
        else:
            break


    print("✅ JSON data imported into PostgreSQL successfully!")

if __name__ == "__main__":
    import_json_to_postgres()
