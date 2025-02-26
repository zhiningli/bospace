from fastapi import APIRouter
from src.api.basemodel import BORequest
from workers.celery_worker import run_bo_task

router = APIRouter()


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
