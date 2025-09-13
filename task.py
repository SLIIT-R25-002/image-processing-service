# tasks.py
# Celery worker that offloads heavy analysis to background

import os
from celery import Celery
from ai_logic import run_analysis_pipeline

# You can override the broker/backend via env vars
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery = Celery(
    "tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

@celery.task(name="tasks.analyze_image_task")
def analyze_image_task(image_path: str):
    """
    Background task: runs the full material analysis pipeline.
    """
    result = run_analysis_pipeline(image_path)
    # Ensure the result is JSON-serializable
    return result
