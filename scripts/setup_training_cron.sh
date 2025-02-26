#!/bin/bash

echo "🚀 Setting up daily training cron job for BOSpace..."

# Get the project root directory
PROJECT_ROOT="/home/zhining/bospace"
echo "📁 Project root detected as: $PROJECT_ROOT"

# Define paths
PYTHON_EXEC="$PROJECT_ROOT/venv/bin/python"  # Ensuring correct env
TRAIN_SCRIPT="$PROJECT_ROOT/scripts/train_daily.py"
LOG_FILE="$PROJECT_ROOT/logs/train_daily.log"

# Ensure the log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Remove existing cron job if it exists
echo "🧹 Cleaning up any existing cron job..."
(crontab -l | grep -v "$TRAIN_SCRIPT") | crontab -

# Add new cron job with Conda environment activation
echo "🕒 Scheduling new weekly training cron job at 2 AM every Thursday..."
CRON_JOB="0 2 * * 4 source ~/miniconda3/etc/profile.d/conda.sh && conda activate bospace && export PYTHONPATH=\"$PROJECT_ROOT/src\" && $PYTHON_EXEC $TRAIN_SCRIPT >> $LOG_FILE 2>&1"
(crontab -l; echo "$CRON_JOB") | crontab -

echo "✅ Cron job successfully set up! Check with: crontab -l"
