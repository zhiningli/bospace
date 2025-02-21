from src.service.hyperparameter_evaluation_service import HPEvalutaionService


sev = HPEvalutaionService()
sev.run_hp_evaluations_for_all_models()