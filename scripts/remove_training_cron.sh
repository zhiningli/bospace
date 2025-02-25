#!/bin/bash

echo "🗑️ Removing daily training cron job..."

# Remove the specific cron job
(crontab -l | grep -v "/home/zhining/bospace/scripts/train_daily.py") | crontab -

echo "✅ Cron job removed successfully!"
