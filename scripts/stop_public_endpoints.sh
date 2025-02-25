#!/bin/bash

echo "🛑 Stopping services for bospace..."

# Stop Uvicorn
echo "🔄 Stopping Uvicorn..."
if pgrep -f "uvicorn main:app" > /dev/null; then
    pkill -f "uvicorn main:app"
    echo "✅ Uvicorn stopped."
else
    echo "⚠️  Uvicorn not running."
fi

# Stop Celery workers
echo "🔄 Stopping Celery workers..."
if pgrep -f "celery -A src.workers.celery_worker" > /dev/null; then
    pkill -f "celery -A src.workers.celery_worker"
    echo "✅ Celery workers stopped."
else
    echo "⚠️  No active Celery workers found."
fi

# Stop Redis
echo "🔄 Stopping Redis..."
if systemctl is-active --quiet redis; then
    sudo systemctl stop redis
    echo "✅ Redis stopped."
else
    echo "⚠️  Redis was not running."
fi

echo "🎉 All services have been stopped successfully."
