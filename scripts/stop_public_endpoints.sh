#!/bin/bash

echo "🛑 Stopping all services for bospace..."

# Stop Uvicorn
echo "🔄 Stopping Uvicorn..."
if pgrep -f "uvicorn main:app" > /dev/null; then
    pkill -15 -f "uvicorn main:app"
    echo "✅ Uvicorn stopped."
else
    echo "⚠️  Uvicorn was not running."
fi

# Stop Celery workers
echo "🔄 Stopping Celery workers..."
if pgrep -f "celery worker" > /dev/null; then
    pkill -15 -f "celery worker"
    echo "✅ Celery workers stopped."
else
    echo "⚠️  No active Celery workers found."
fi

# Stop Redis (Optional: only stop if you want a full shutdown)
echo "🔄 Stopping Redis..."
if systemctl is-active --quiet redis; then
    redis-cli DEL bo_task_updates
    echo "✅ Deleted remaining key value store"
    sudo systemctl stop redis
    echo "✅ Redis stopped."
else
    echo "⚠️  Redis was not running."
fi

# Stop orphaned background processes (just in case)
echo "🔄 Checking for any remaining background processes..."
pkill -f "uvicorn"
pkill -f "celery"
echo "✅ All remaining background processes stopped."

echo "🎉 All services have been stopped successfully."
