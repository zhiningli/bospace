from celery import Celery

celery_app = Celery(
    "bospace",
    broker="redis://localhost:6379/0",  # Ensure this matches your Redis setup
    backend=None
)
# Celery configuration
celery_app.conf.update(
    tast_track_started = True,
    result_expires=3600,
    worker_prefetch_multiplier=1,
    task_acks_late=True
)

