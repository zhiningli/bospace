from src.config.celery_config import celery_app
from src.utils import convert_bo_output_to_json_format
from src.config import redis_client
from src.service import BOService, SimilarityInferenceService
import logging
import threading
import time


logger = logging.getLogger("service")

@celery_app.task(name="src.workers.celery_worker.run_bo_task")
def run_bo_task(code_string: str, search_space: dict, n_initial_points, n_iter, allow_logging):
    """Background task for Bayesian Optimisation with Redis Stream updates."""

    task_id = run_bo_task.request.id
    stream_name = "bo_task_updates"

    try:
        logger.info("Starting Bayesian Optimisation task")
        redis_client.xadd(stream_name, {"task_id": task_id, "status": "STARTED"})

        bo_service = BOService()
        search_space = bo_service.load_constrained_search_space(search_space)

        start_time = time.time() 

        def send_progress_updates():
            while True:
                time.sleep(30) 
                elapsed_time = time.time() - start_time 
                
                redis_client.xadd(stream_name, {
                    "task_id": task_id,
                    "task_type": "Bayesian Optimisation",
                    "status": "STILL_PROCESSING",
                    "elapsed_time": round(elapsed_time, 2) 
                })

                active_tasks = celery_app.control.inspect().active()

                if not active_tasks:
                    break

                active_task_ids = [
                    task["id"] for worker_tasks in active_tasks.values() for task in worker_tasks
                ]

                if task_id not in active_task_ids:
                    break




        progress_thread = threading.Thread(target=send_progress_updates, daemon=True)
        progress_thread.start()

        accuracies, best_y, best_candidate = bo_service.optimise(
            code_str=code_string,
            search_space=search_space,
            initial_points=n_initial_points,
            n_iter=n_iter
        )

        result = convert_bo_output_to_json_format(constrained_search_space=search_space,accuracies=accuracies,best_y=best_y,best_candidates=best_candidate)
        total_runtime = round(time.time() - start_time, 2)
        redis_client.xadd(stream_name, {"task_id": task_id, "status": "COMPLETED", "result": str(result), "total_time": str(total_runtime)})

        return result

    except Exception as e:
        logger.error(f"BO Task failed: {e}")
        redis_client.xadd(stream_name, {"task_id": task_id, "status": "FAILED", "error": str(e)})
        raise RuntimeError(f"BO Task failed: {str(e)}")

    

@celery_app.task(name="src.workers.celery_worker.suggest_search_space_task")
def suggest_search_space_task(code_string: str, top_k: int=5):
    task_id = suggest_search_space_task.request.id
    stream_name = "search_space_update"

    try:
        logger.info("Starting suggesting search space task")
        redis_client.xadd(stream_name, {"task_id": task_id, "status": "STARTED"})

        service = SimilarityInferenceService()
        search_space = service.suggest_search_space(
            code_str=code_string,
            num_similar_dataset=top_k,
            num_similar_model=top_k
        )

        logger.info(f"Search space suggestion completed successfully: {search_space}")
        redis_client.xadd(stream_name, {"task_id": task_id, "status": "COMPLETED", "result": str(search_space)})
        return search_space
    
    except Exception as e:
        logger.error(f"Suggest search space task failed: {e}", exc_info=True)
        redis_client.xadd(stream_name, {"task_id": task_id, "status": "FAILED", "error": str(e)})
        raise RuntimeError(f"Suggest search space task failed: {str(e)}")
    


# @celery_app.task(name="src.workers.celery_worker.run_bo_task")
# def run_bo_task(code_string, search_space, n_initial_points, n_iter, allow_logging):
#     """Background task for Bayesian Optimisation"""
#     try:
#         logger.info("Starting Bayesian Optimisation task.")

#         bo_service = BOService()
#         search_space = bo_service.load_constrained_search_space(search_space)
#         accuracies, best_y, best_candidate = bo_service.optimise(
#             code_str = code_string,
#             search_space = search_space,
#             initial_points = n_initial_points,
#             n_iter=n_iter 
#         )
#         logger.info(f"Optimization completed successfully: {accuracies, best_y, best_candidate}")
        
#         if allow_logging:
#             #TODO implement logging, 
#             # save model to the model table
#             # save database to the database table if allowed
#             # save script and result to the script table
#             # So that I can reuse the results in the future
#             pass
#         result = convert_bo_output_to_json_format(search_space, accuracies, best_y, best_candidate)

#         return result
    
#     except Exception as e:
#         logger.error(f"BO Task failed: {e}")
#         raise RuntimeError(f"BO Task failed: {str(e)}")

# @celery_app.task(name="src.workers.celery_worker.suggest_search_space_task")
# def suggest_search_space_task(code_string: str, top_k: int =5):
#     """Background task to suggest search space on model similarities"""
#     try:
#         logger.info("Statting suggesting search space task")
#         from src.service.similarity_inference_service import SimilarityInferenceService
#         service = SimilarityInferenceService()
#         search_space = service.suggest_search_space(code_str=code_string, num_similar_dataset=top_k, num_similar_model=top_k)

#         return search_space
#     except Exception as e:
#         logger.error(f"Suggest search space task fails: {e}", exc_info=True)
#         raise RuntimeError(f"Suggest search space task failed: {str(e)}")