#!/bin/bash

# Configuration
PROJECT_DIR="/home/zhining/bospace"
PYTHON_ENV="bospace"
REDIS_PORT=6379

echo "Starting project setup..."

# Activate Conda Environment
echo "Activating Conda environment: $PYTHON_ENV"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $PYTHON_ENV

# Set Environment Variables
echo "Setting PYTHONPATH..."
export PYTHONPATH=$PROJECT_DIR/src

# Start Redis if not running
echo "Checking Redis status..."
if ! nc -z localhost $REDIS_PORT; then
    echo "Starting Redis..."
    sudo systemctl start redis
else
    echo "Redis is already running on port $REDIS_PORT."
fi

# Purge old Celery tasks (Optional)
echo "Purging old Celery tasks..."
celery -A src.workers.celery_worker purge --force

# Start Celery Worker
echo "Starting Celery worker..."
gnome-terminal -- bash -c "cd $PROJECT_DIR && conda activate $PYTHON_ENV && celery -A src.workers.celery_worker worker --loglevel=info; exec bash"

# Start FastAPI App
echo "Starting FastAPI app with Uvicorn..."
gnome-terminal -- bash -c "cd $PROJECT_DIR && conda activate $PYTHON_ENV && uvicorn main:app --reload --host 0.0.0.0 --port 8000 --limit-max-requests 1000; exec bash"

echo "All services started successfully! API listening on http://localhost:8000/"
