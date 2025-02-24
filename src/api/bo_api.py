from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from workers.celery_worker import run_bo_task
from celery.result import AsyncResult

router = APIRouter()

class BOSearchSpace(BaseModel):
    learning_rate: tuple[float, float] = Field(..., description="Bounds for learning rate.")
    momentum: tuple[float, float]=Field(..., description="Bounds for momentum.")
    weight_decay: tuple[float, float]=Field(..., description="Bounds for weight decay")
    num_epochs: tuple[float, float]=Field(..., description="Bounds for number of epochs")

class BORequest(BaseModel):
    code_str: str
    search_space: BOSearchSpace
    n_initial_points: int = 5
    n_iter: int = 20
    allow_logging: bool = True

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
