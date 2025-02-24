from celery import Celery


# Celery instance
celery_app = Celery(
    "bospace",
    broker = "redis://localhost:6379/0",
    backend = "redis://localhost:6379/0"
)

# Celery configuration
celery_app.conf.update(
    tast_track_started = True,
    result_expires=3600
)

