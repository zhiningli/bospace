import json
from src.database.object.result import Result
from src.database.crud.result_crud import ResultRepository
from src.database.crud import ScriptRepository

# Path to your exported JSON file
json_file_path = "scripts.json"

repo = ScriptRepository()

def extract_sgd_config(entry, result_type):
    """Extract the best performing configuration from JSON."""
    result = entry.get(result_type, {})
    best_hyperparameters = result.get("best_hyperparameters", {})
    return {
        "learning_rate": best_hyperparameters.get("learning_rate", 0.0),
        "momentum": best_hyperparameters.get("momentum", 0.0),
        "weight_decay": best_hyperparameters.get("weight_decay", 0.0),
        "num_epochs": best_hyperparameters.get("num_epochs", 0),
        "highest_score": result.get("best_score"),
    }

def import_json_to_postgres():
    """Import JSON data into PostgreSQL."""
    with open(json_file_path, "r") as file:
        data = json.load(file)

    for entry in data:
        dataset_idx = entry["dataset_idx"]
        model_idx = entry["model_idx"]
        script_idx = repo.get_script_id_by_model_idx_dataset_idx(model_idx=model_idx, dataset_idx=dataset_idx)

        if "unconstrained_result" in entry.keys():
            result = Result(
                script_idx= script_idx,
                model_idx= model_idx,
                dataset_idx= dataset_idx,
                result_type = "unconstrained",
                sgd_best_performing_configuration=extract_sgd_config(entry, "unconstrained_result")
            )
        # ✅ 2. Use DatasetRepository to insert into DB
            inserted_result = ResultRepository.create_result(
                script_idx = result.script_idx,
                model_idx = result.model_idx,
                dataset_idx = result.dataset_idx,
                result_type= result.result_type,
                sgd_config= result.sgd_best_performing_configuration
            )

            if inserted_result:
                print(f"✅ Successfully inserted models {inserted_result.result_id}")
            else:
                break

        if "constrained_search_space" in entry.keys():
            result = Result(
                script_idx= script_idx,
                model_idx= model_idx,
                dataset_idx= dataset_idx,
                result_type = "constrained",
                sgd_best_performing_configuration=extract_sgd_config(entry, "constrained_search_space")
            )
        # ✅ 2. Use DatasetRepository to insert into DB
            inserted_result = ResultRepository.create_result(
                script_idx = result.script_idx,
                model_idx = result.model_idx,
                dataset_idx = result.dataset_idx,
                result_type= result.result_type,
                sgd_config= result.sgd_best_performing_configuration
            )

            if inserted_result:
                print(f"✅ Successfully inserted models {inserted_result.result_id}")
            else:
                break

        if "improved_constrained_search_space" in entry.keys():
            result = Result(
                script_idx= script_idx,
                model_idx= model_idx,
                dataset_idx= dataset_idx,
                result_type = "improved_constrained",
                sgd_best_performing_configuration=extract_sgd_config(entry, "improved_constrained_search_space")
            )
        # ✅ 2. Use DatasetRepository to insert into DB
            inserted_result = ResultRepository.create_result(
                script_idx = result.script_idx,
                model_idx = result.model_idx,
                dataset_idx = result.dataset_idx,
                result_type= result.result_type,
                sgd_config= result.sgd_best_performing_configuration
            )

            if inserted_result:
                print(f"✅ Successfully inserted models {inserted_result.result_id}")
            else:
                break

        if "constrained_unseen_search_space" in entry.keys():
            result = Result(
                script_idx= script_idx,
                model_idx= model_idx,
                dataset_idx= dataset_idx,
                result_type = "unseen_constrained",
                sgd_best_performing_configuration=extract_sgd_config(entry, "constrained_unseen_search_space")
            )
        # ✅ 2. Use DatasetRepository to insert into DB
            inserted_result = ResultRepository.create_result(
                script_idx = result.script_idx,
                model_idx = result.model_idx,
                dataset_idx = result.dataset_idx,
                result_type= result.result_type,
                sgd_config= result.sgd_best_performing_configuration
            )

            if inserted_result:
                print(f"✅ Successfully inserted models {inserted_result.result_id}")
            else:
                break

        if "improved_unseen_constrained_search_space" in entry.keys():
            result = Result(
                script_idx= script_idx,
                model_idx= model_idx,
                dataset_idx= dataset_idx,
                result_type = "improved_unseen_constrained",
                sgd_best_performing_configuration=extract_sgd_config(entry, "improved_unseen_constrained_search_space")
            )
        # ✅ 2. Use DatasetRepository to insert into DB
            inserted_result = ResultRepository.create_result(
                script_idx = result.script_idx,
                model_idx = result.model_idx,
                dataset_idx = result.dataset_idx,
                result_type= result.result_type,
                sgd_config= result.sgd_best_performing_configuration
            )

            if inserted_result:
                print(f"✅ Successfully inserted models {inserted_result.result_id}")
            else:
                break


    print("✅ JSON data imported into PostgreSQL successfully!")

if __name__ == "__main__":
    import_json_to_postgres()
