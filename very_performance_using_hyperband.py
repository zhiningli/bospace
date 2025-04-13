from src.service import SimilarityInferenceService, HyperbandService
from src.database import ScriptRepository, ResultRepository, ModelRepository, DatasetRepository
from src.assets import code_str
from src.assets import sgd_search_space
import json

similarity_service = SimilarityInferenceService()
hyperband_service = HyperbandService(similarity_service.store)

dataset_with_codes = DatasetRepository.get_all_dataset()
dataset_codes_dict = {int(datasetCode.dataset_idx): datasetCode for datasetCode in dataset_with_codes}

results = {}

for model_idx in range(1, 16):
    for dataset_idx in range(1, 15):    
        dataset_code = dataset_codes_dict[dataset_idx].code
        input_size = dataset_codes_dict[dataset_idx].input_size
        num_classes = dataset_codes_dict[dataset_idx].num_classes
        model_code = ModelRepository.get_model_with_code(model_idx=model_idx).code
        
        script_code = code_str.format(dataset=dataset_code, model=model_code, input_size=input_size, num_classes=num_classes)

        print(f"Running UNCONSTRAINED Hyperband for model {model_idx}, dataset {dataset_idx}")
        unconstrained_accuracies, unconstrained_best_config, unconstrained_best_score, unconstrained_score_log = hyperband_service.optimise(
            code_str=script_code,
            search_space=sgd_search_space,
            return_score_log=True
        )

        print(f"Commencing CONSTRAINED search space suggestion...")
        old_script = ScriptRepository.get_script_by_model_and_dataset_idx(model_idx=model_idx, dataset_idx=dataset_idx)
        script_idx = old_script.script_idx
        original_sgd_performance = old_script.sgd_best_performing_configuration

        new_search_space = similarity_service.suggest_search_space(code_str=script_code, target_model_idx=model_idx, target_dataset_idx=dataset_idx)
        constrained_search_space = hyperband_service.load_constrained_search_space(search_space=new_search_space)

        print(f"Running CONSTRAINED Hyperband for model {model_idx}, dataset {dataset_idx}")
        constrained_accuracies, constrained_best_config, constrained_best_score, constrained_score_log = hyperband_service.optimise(
            code_str=script_code,
            search_space=constrained_search_space,
            return_score_log=True
        )

        if constrained_best_score >= original_sgd_performance["highest_score"]:
            ScriptRepository.update_script_sgd_config(script_idx=script_idx, sgd_config=constrained_best_config)

        results[f"{model_idx}_{dataset_idx}"] = {
            "unconstrained_results": {
                "best_score": unconstrained_best_score,
                "best_config": unconstrained_best_config,
                "score_timeline": unconstrained_score_log
            },
            "constrained_results": {
                "best_score": constrained_best_score,
                "best_config": constrained_best_config,
                "score_timeline": constrained_score_log
            }
        }

        with open("evaluation_results_hyperband.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved for model {model_idx}, dataset {dataset_idx} ✅")

