import logging
from src.service import SimilarityTrainingService

logging.basicConfig(filename='/home/zhining/bospace/logs/train_daily.log', level=logging.INFO)

def main():
    logging.info("🚀 [TRAINING STARTED] Running daily training task...")
    try:
        service = SimilarityTrainingService()
        service.prepare_data_for_rank_training()
        service.training_RFRegressor_for_dataset_rank_prediction()
        service.training_XGBoost_for_model_rank_prediction()
        logging.info("✅ [TRAINING COMPLETED] Daily training task completed successfully!")
    except Exception as e:
        logging.error(f"❌ [TRAINING FAILED] Error: {e}")

if __name__ == "__main__":
    main()
