#!/bin/bash

echo "🚀 Starting project setup..."

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "📁 Project root detected as: $PROJECT_ROOT"

# Activate Conda environment
echo "🟢 Activating Conda environment: bospace"
source ~/miniconda3/etc/profile.d/conda.sh || { echo "❌ Conda not found!"; exit 1; }
conda activate bospace || { echo "❌ Failed to activate Conda environment!"; exit 1; }

# Set PYTHONPATH relative to project root
echo "🌍 Setting PYTHONPATH..."
export PYTHONPATH="$PROJECT_ROOT/src"
echo "PYTHONPATH set to $PYTHONPATH"

# Check Redis status
echo "🔄 Checking Redis status..."
if ! systemctl is-active --quiet redis; then
    echo "🚀 Starting Redis..."
    sudo systemctl start redis || { echo "❌ Failed to start Redis!"; exit 1; }
else
    echo "✅ Redis is already running on port 6379."
fi

# Purge old Celery tasks
echo "🗑️  Purging old Celery tasks..."
celery -A src.workers.celery_worker purge -f || echo "⚠️  Failed to purge Celery tasks (may be empty)."

echo "⚙️  Starting Celery worker..."
(cd $PROJECT_ROOT && celery -A src.workers.celery_worker worker --loglevel=info &)

# Start Uvicorn for FastAPI app in background
echo "🌐 Starting FastAPI app with Uvicorn..."
(cd $PROJECT_ROOT && uvicorn main:app --reload --host 0.0.0.0 --port 8000 --limit-max-requests 1000 &) 

echo "🎉 All services started successfully! Service listening on http://localhost:8000/"
