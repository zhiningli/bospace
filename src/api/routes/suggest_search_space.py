from fastapi import APIRouter
from src.api.basemodel import SuggestSearchSpaceRequest
from workers.celery_worker import suggest_search_space_task

router = APIRouter()

@router.post("/suggest-search-space")
async def start_suggest_search_space(request: SuggestSearchSpaceRequest):
    task = suggest_search_space_task.apply_async(args=[request.code_str, request.top_k])
    return {
        "task_id": task.id,
        "status": "Task started_successfully"
    }


