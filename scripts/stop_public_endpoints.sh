#!/bin/bash

echo "Stopping services for bospace..."

# Kill Uvicorn process
echo "Stopping Uvicorn..."
pkill -f "uvicorn main:app"

# Stop Celery workers
echo "Stopping Celery workers..."
pkill -f "celery"

# Stop Redis
echo "Stopping Redis..."
sudo systemctl stop redis

echo "All services have been stopped."
