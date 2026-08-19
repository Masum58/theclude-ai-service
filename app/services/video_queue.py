# WHAT: Redis/RQ queue access for video-processing jobs
# WHY: Upload requests should enqueue heavy video work instead of running Whisper and FFmpeg
#      inside the FastAPI process.
# INPUT: video_id and video_url from the upload route
# OUTPUT: an RQ job that a separate worker will process
# WHO CALLS: app.api.routes.video
# CONNECTED TO: Redis, RQ worker, app.tasks.video_processing

from redis import Redis
from rq import Queue

from app.core.config import settings


def get_video_queue() -> Queue:
    """Return the configured Redis queue for video-processing work."""
    connection = Redis.from_url(settings.REDIS_URL)
    return Queue(name=settings.VIDEO_QUEUE_NAME, connection=connection)


def enqueue_video_processing(video_id: str, video_url: str):
    """Place one video pipeline invocation on Redis and return its RQ job."""
    queue = get_video_queue()
    return queue.enqueue(
        "app.tasks.video_processing.run_video_processing",
        video_id,
        video_url,
        job_timeout=settings.VIDEO_JOB_TIMEOUT_SECONDS,
    )
