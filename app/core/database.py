# WHAT: Minimal PostgreSQL connection and schema setup for the AI microservice
# WHY: Video-processing job records must survive API restarts and multiple server workers.
# INPUT: DATABASE_URL from configuration
# OUTPUT: PostgreSQL connections and the video_processing_jobs table
# WHO CALLS: app.main at startup and app.services.job_tracker for each query

import psycopg

from app.core.config import settings


def get_connection():
    """Open a short-lived autocommit PostgreSQL connection for one database operation."""
    return psycopg.connect(settings.DATABASE_URL, autocommit=True)


def initialize_database() -> None:
    """Create the persistent job-status table when the AI service starts."""
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS video_processing_jobs (
                    video_id VARCHAR(100) PRIMARY KEY,
                    status VARCHAR(20) NOT NULL CHECK (status IN ('queued', 'processing', 'completed', 'failed')),
                    chunks_stored INTEGER,
                    error_message TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cursor.execute(
                """
                ALTER TABLE video_processing_jobs
                DROP CONSTRAINT IF EXISTS video_processing_jobs_status_check
                """
            )
            cursor.execute(
                """
                ALTER TABLE video_processing_jobs
                ADD CONSTRAINT video_processing_jobs_status_check
                CHECK (status IN ('queued', 'processing', 'completed', 'failed'))
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_video_processing_jobs_status
                ON video_processing_jobs (status)
                """
            )
