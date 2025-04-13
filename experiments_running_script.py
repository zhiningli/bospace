# This script is for continuing adding better information to the result table
from src.service import SimilarityInferenceService, BOService
from src.database import ScriptRepository, ResultRepository, ModelRepository, DatasetRepository
from src.assets import code_str

similarity_service = SimilarityInferenceService()
bo_service = BOService(similarity_service.store)
dataset_with_codes = DatasetRepository.get_all_dataset()
dataset_codes_dict = {int(datasetCode.dataset_idx): datasetCode for datasetCode in dataset_with_codes}

for i in range(1):
    for model_idx in range(25, 30):
        for dataset_idx in range(16, 26):    
            print(f"Commencing search space suggestion for dataset {dataset_idx}, model {model_idx} script")
            
            dataset_code = dataset_codes_dict[dataset_idx].code
            input_size = dataset_codes_dict[dataset_idx].input_size
            num_classes = dataset_codes_dict[dataset_idx].num_classes
            model_code = ModelRepository.get_model_with_code(model_idx=model_idx).code

            script_code = code_str.format(dataset=dataset_code, model=model_code, input_size=input_size, num_classes=num_classes)

            old_script = ScriptRepository.get_script_by_model_and_dataset_idx(model_idx=model_idx, dataset_idx=dataset_idx)
            
            script_idx = old_script.script_idx
            original_sgd_performance = old_script.sgd_best_performing_configuration

            search_space = similarity_service.suggest_search_space(code_str = script_code, target_model_idx = model_idx, target_dataset_idx = dataset_idx)
            print(search_space)
            constrained_search_space = bo_service.load_constrained_search_space(search_space=search_space)
            print(f"Search space suggested, running bayesian optimsation")
            accuracies, best_y, best_candidate =  bo_service.optimise(code_str=script_code, search_space=constrained_search_space, n_iter=20, initial_points=12)

            new_best_candidate = list(map(int, best_candidate.flatten().tolist()))

            best_sgd_configurations = {
                "learning_rate": constrained_search_space["learning_rate"][int(new_best_candidate[0])],
                "momentum": constrained_search_space["momentum"][int(new_best_candidate[1])],
                "weight_decay": constrained_search_space["weight_decay"][int(new_best_candidate[2])],
                "num_epochs": constrained_search_space["num_epochs"][int(new_best_candidate[3])],
                "highest_score": float(best_y)
                }
            
            if float(best_y) >= original_sgd_performance["highest_score"]:
                ScriptRepository.update_script_sgd_config(script_idx=script_idx, sgd_config=best_sgd_configurations)

            ResultRepository.create_result(
                script_idx=script_idx,
                model_idx=model_idx,
                dataset_idx=dataset_idx,
                result_type="ktrc_inferred_constrained",
                sgd_config=best_sgd_configurations
            )
            
            print(f"Results for dataset {dataset_idx}, model {model_idx} script saved successfully! Iteration {i+1}")

