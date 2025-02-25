from src.config.celery_config import celery_app
from src.utils import convert_bo_output_to_json_format

import logging

logger = logging.getLogger("service")


@celery_app.task(name="src.workers.celery_worker.run_bo_task")
def run_bo_task(code_string, search_space, n_initial_points, n_iter, allow_logging):
    """Background task for Bayesian Optimisation"""
    try:
        logger.info("Starting Bayesian Optimisation task.")
        from src.service import BOService
        bo_service = BOService()
        search_space = bo_service.load_constrained_search_space(search_space)
        accuracies, best_y, best_candidate = bo_service.optimise(
            code_str = code_string,
            search_space = search_space,
            initial_points = n_initial_points,
            n_iter=n_iter 
        )
        logger.info(f"Optimization completed successfully: {accuracies, best_y, best_candidate}")
        
        if allow_logging:
            #TODO implement logging, 
            # save model to the model table
            # save database to the database table if allowed
            # save script and result to the script table
            # So that I can reuse the results in the future
            pass
        result = convert_bo_output_to_json_format(search_space, accuracies, best_y, best_candidate)

        return result
    
    except Exception as e:
        logger.error(f"BO Task failed: {e}")
        raise RuntimeError(f"BO Task failed: {str(e)}")

@celery_app.task(name="src.workers.celery_worker.suggest_search_space_task")
def suggest_search_space_task(self, code_string: str, top_k: int =5):
    """Background task to suggest search space on model similarities"""
    try:
        logger.info("Statting suggesting search space task")
        from src.service.similarity_inference_service import SimilarityInferenceService
        service = SimilarityInferenceService()

        search_space = service.suggest_search_space(code_str=code_string, num_similar_dataset=top_k, num_similar_model=top_k)

        return search_space
    except Exception as e:
        logger.error(f"Suggest search space task fails: {e}", exc_info=True)
        raise RuntimeError(f"Suggest search space task failed: {str(e)}")