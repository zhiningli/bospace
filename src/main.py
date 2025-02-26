from fastapi import FastAPI
from src.api.routes import bo_router, search_space_router, task_update_router

app = FastAPI()

# Include API routes
app.include_router(bo_router, tags=["Bayesian Optimisation"])
app.include_router(search_space_router, tags=["Suggest Search Space"])
app.include_router(task_update_router, prefix="/updates", tags = ["Task Updates"])

# Optional: Root endpoint
@app.get("/")
async def read_root():
    return {"message": "API is running!"}
