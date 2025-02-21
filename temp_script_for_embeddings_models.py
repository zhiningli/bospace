from src.service.embeddings import Model_Embedder
from src.database.crud import ModelRepository
from src.preprocessing.preprocessor import Tokeniser
import torch

models = ModelRepository.get_all_models()
preprocessor = Tokeniser()
embedder = Model_Embedder()

for model in models:
    model_idx = model.model_idx
    model_code = model.code
    tokens = preprocessor(model_code)
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    input_ids = torch.tensor([input_ids]) 
    attention_mask = torch.tensor([attention_mask])
    feature_vector = embedder.get_embedding(input_ids=input_ids, attention_mask=attention_mask).view(-1).tolist()

    ModelRepository.update_feature_vector(model_idx=model_idx, feature_vector=feature_vector)
    print(f"Feature vector for model {model_idx} updated successfully!")
