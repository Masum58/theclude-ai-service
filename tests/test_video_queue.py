# WHAT: Unit tests for the Redis/RQ video queue
# WHY: Verify uploads are enqueued as worker jobs without connecting to Redis.
# INPUT: Mocked RQ queue
# OUTPUT: Verified task target, payload, and timeout

from unittest.mock import MagicMock, patch

from app.core.config import settings
from app.services.video_queue import enqueue_video_processing


@patch("app.services.video_queue.get_video_queue")
def test_enqueue_video_processing_uses_configured_worker_task(mock_get_queue):
    queue = MagicMock()
    mock_get_queue.return_value = queue

    enqueue_video_processing("video-123", "https://example.com/lesson.mp4")

    queue.enqueue.assert_called_once_with(
        "app.tasks.video_processing.run_video_processing",
        "video-123",
        "https://example.com/lesson.mp4",
        job_timeout=settings.VIDEO_JOB_TIMEOUT_SECONDS,
    )
