from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from service.celery_worker import run_bo_task
from celery.result import AsyncResult

router = APIRouter()

class BOSearchSpace(BaseModel):
    pass