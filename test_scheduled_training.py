from src.service import SimilarityTrainingService

service = SimilarityTrainingService()
service.prepare_data_for_rank_training()
service.training_RFRegressor_for_dataset_rank_prediction()
service.training_XGBoost_for_model_rank_prediction()