from src.database import DatasetRepository, ModelRepository
from datetime import datetime
import numpy as np
from src.service import SimilarityTrainingService, SimilarityInferenceService
from src.model import DatasetSimilarityModel
import os
from dotenv import load_dotenv


from sklearn.model_selection import train_test_split

training_service = SimilarityTrainingService()

training_service.training_for_dataset_similarity()

load_dotenv()

dataset_similarity_file = os.path.join(os.getenv("DATASET_RANK_PREDICTION_MODEL_PATH", "./"), "dataset_similarity.pkl")
dataset_predictor = DatasetSimilarityModel()
dataset_predictor.load_model(dataset_similarity_file)
# Inference
datasets = DatasetRepository.get_all_dataset()

count = 0
for dataset in datasets:
    dataset_idx = dataset.dataset_idx
    dataset_meta_features = dataset.meta_features
    res = []
    for source_dataset in datasets:
        source_dataset_idx = source_dataset.dataset_idx
        source_dataset_meta_features = source_dataset.meta_features

        features = np.array(dataset_meta_features + source_dataset_meta_features).reshape(1, -1)

        score = dataset_predictor.predict(features)

        res.append((source_dataset_idx, score))
    print(len(res))
    res.sort(key= lambda x:x[1], reverse=True)
    print("dataset_idx", source_dataset_idx)
    print(res[:10])
    output =  [res[i][0] for i in range(5)]

    if dataset_idx in output:
        count += 1
        print("Success for model", dataset_idx)
print(count)


