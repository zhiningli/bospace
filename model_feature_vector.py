from src.service import Model_Embedder
from src.database import ModelRepository

embedder = Model_Embedder()

models = ModelRepository.get_all_models()

for model in models:

    model_idx = model.model_idx
    code = model.code
    
    new_feature_vector = embedder.get_embedding(code)

    ModelRepository.update_feature_vector(model_idx=model_idx, feature_vector=new_feature_vector)