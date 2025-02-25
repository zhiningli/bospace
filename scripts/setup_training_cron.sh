#!/bin/bash

echo "🚀 Setting up daily training cron job for BOSpace..."

# Define paths
PYTHON_EXEC="/home/zhining/miniconda3/envs/bospace/bin/python"
TRAIN_SCRIPT="/home/zhining/bospace/scripts/train_daily.py"
LOG_FILE="/home/zhining/bospace/logs/train_daily.log"

# Ensure the log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Remove existing cron job if exists
echo "🧹 Cleaning up any existing cron job..."
(crontab -l | grep -v "$TRAIN_SCRIPT") | crontab -

# Add new cron job
echo "🕒 Scheduling new daily training cron job at 2 AM..."
echo "0 2 * * * $PYTHON_EXEC $TRAIN_SCRIPT >> $LOG_FILE 2>&1" | crontab -

echo "✅ Cron job successfully set up! Check with: crontab -l"
