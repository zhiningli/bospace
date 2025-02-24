import uvicorn

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", post=8000, reload=True)