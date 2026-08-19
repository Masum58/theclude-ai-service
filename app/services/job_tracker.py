# WHAT: PostgreSQL-backed tracking for video-processing jobs
# WHY: The processing graph runs in the background, so status must persist beyond a server
#      restart and be shared by every API worker.
# INPUT: video_id, status updates, optional chunks_stored/error_message
# OUTPUT: the current status record for a video, or None when it has never been submitted
# WHO CALLS: app.api.routes.video and app.api.routes.video_qa
# CONNECTED TO: app.core.database and the video_processing_jobs PostgreSQL table

from typing import Optional, TypedDict

from app.core.database import get_connection


class JobRecord(TypedDict):
    """The status information returned for one video-processing job."""

    status: str
    chunks_stored: Optional[int]
    error_message: Optional[str]


def _upsert_job(
    video_id: str,
    status: str,
    chunks_stored: Optional[int] = None,
    error_message: Optional[str] = None,
) -> None:
    """Create a job record or replace its latest processing state using raw SQL."""
    if not video_id or not video_id.strip():
        raise ValueError("video_id cannot be empty.")

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO video_processing_jobs
                    (video_id, status, chunks_stored, error_message)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (video_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    chunks_stored = EXCLUDED.chunks_stored,
                    error_message = EXCLUDED.error_message,
                    updated_at = NOW()
                """,
                (video_id, status, chunks_stored, error_message),
            )


def create_job_queued(video_id: str) -> bool:
    """Create a queued job only if this video has never been submitted before."""
    if not video_id or not video_id.strip():
        raise ValueError("video_id cannot be empty.")

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO video_processing_jobs (video_id, status, chunks_stored, error_message)
                VALUES (%s, 'queued', NULL, NULL)
                ON CONFLICT (video_id) DO NOTHING
                RETURNING video_id
                """,
                (video_id,),
            )
            return cursor.fetchone() is not None


def set_job_processing(video_id: str) -> None:
    """Mark a queued video as processing when its Redis worker starts."""
    _upsert_job(video_id=video_id, status="processing")


def set_job_completed(video_id: str, chunks_stored: int) -> None:
    """Mark a video as completed and store the number of indexed transcript chunks."""
    _upsert_job(video_id=video_id, status="completed", chunks_stored=chunks_stored)


def set_job_failed(video_id: str, error_message: str) -> None:
    """Mark a video as failed and retain the pipeline error for status clients."""
    _upsert_job(video_id=video_id, status="failed", error_message=error_message)


def get_job_status(video_id: str) -> Optional[JobRecord]:
    """Return a persisted job record, or None if this video has never been submitted."""
    if not video_id or not video_id.strip():
        raise ValueError("video_id cannot be empty.")

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT status, chunks_stored, error_message
                FROM video_processing_jobs
                WHERE video_id = %s
                """,
                (video_id,),
            )
            row = cursor.fetchone()

    if row is None:
        return None

    return {
        "status": row[0],
        "chunks_stored": row[1],
        "error_message": row[2],
    }
