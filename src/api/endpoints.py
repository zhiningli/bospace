from fastapi import APIRouter
from pydantic import BaseModel, Field
from workers.celery_worker import run_bo_task, suggest_search_space_task
from celery.result import AsyncResult

router = APIRouter()


@router.post("/suggest-search-space")
async def start_suggest_search_space(request: SuggestSearchSpaceRequest):
    task = suggest_search_space_task.apply_async(args=[request.code_str, request.top_k])
    return {
        "task_id": task.id,
        "status": "Task started_successfully"
    }

@router.post("/boptimise")
async def start_bayesian_optimisation(request: BORequest):
    task = run_bo_task.apply_async(
        args = [
            request.code_str,
            request.search_space.model_dump(),
            request.n_initial_points,
            request.n_iter,
            request.allow_logging
        ]
    )
    return {"task_id": task.id, "status": "Task started successfully"}


@router.get("/task/{task_id}")
async def get_task_status(task_id: str):
    task_result = AsyncResult(task_id)

    if task_result.state == "PENDING":
        return {"task_id": task_id, "status": "Pending"}

    elif task_result.state == "SUCCESS":
        return {"task_id": task_id, "status": "Completed", "result": task_result.result}
    
    elif task_result.state == "FAILURE":
        return {"task_id": task_id, "status": "Failed", "error": str(task_result.result)}
    
    else:
        return {"task_id": task_id, "status": task_result.state}
