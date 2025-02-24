import uvicorn
from src.service import SimilarityTrainingService

if __name__ == "__main__":

    service = SimilarityTrainingService()

    service.prepare_data_for_rank_training()
    #  uvicorn.run("api:app", host="0.0.0.0", post=8000, reload=True)