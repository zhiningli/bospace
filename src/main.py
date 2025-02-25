from fastapi import FastAPI
from src.api import router

app = FastAPI()

# Include API routes
app.include_router(router)

# Optional: Root endpoint
@app.get("/")
async def read_root():
    return {"message": "API is running!"}
