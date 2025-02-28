from src.service import SimilarityTrainingService, SimilarityInferenceService
from src.database import ModelRepository


service = SimilarityInferenceService()

models = ModelRepository.get_all_models_with_code_string_only()
count = 0
for model in models:
    target_model_idx = model.model_idx
    target_code = model.code
    print("Finding similar models to model", target_model_idx)
    res = service.compute_top_k_model_similarities(model_source_code=target_code, k = 5)
    print(res)
    if target_model_idx in res:
        count += 1
        print("Model idx", target_model_idx, "worked")

print(count)
print("\n")

