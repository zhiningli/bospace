import joblib
from dotenv import load_dotenv
import os

load_dotenv()

file_path = os.getenv("DATASET_RANK_PREDICTION_MODEL_PATH")
# Load the model
model = joblib.load(f"/{file_path}/best_random_forest_model.pkl")

# Print the model's hyperparameters
print(model)
