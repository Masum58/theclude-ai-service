# WHAT: Unit tests for PostgreSQL-backed job tracking
# WHY: Verify the raw SQL tracker sends the correct state updates without requiring Docker.
# INPUT: Mock PostgreSQL connection/cursor objects
# OUTPUT: Verified status records and SQL parameters

from unittest.mock import MagicMock, patch

import pytest

from app.services import job_tracker


def _mock_database(fetchone_result=None):
    connection = MagicMock()
    cursor = MagicMock()
    connection.__enter__.return_value = connection
    connection.cursor.return_value.__enter__.return_value = cursor
    cursor.fetchone.return_value = fetchone_result
    return connection, cursor


@patch("app.services.job_tracker.get_connection")
def test_set_job_completed_persists_completed_state(mock_get_connection):
    connection, cursor = _mock_database()
    mock_get_connection.return_value = connection

    job_tracker.set_job_completed("video-123", chunks_stored=8)

    sql, params = cursor.execute.call_args.args
    assert "INSERT INTO video_processing_jobs" in sql
    assert "ON CONFLICT (video_id) DO UPDATE" in sql
    assert params == ("video-123", "completed", 8, None)


@patch("app.services.job_tracker.get_connection")
def test_create_job_queued_returns_true_when_inserted(mock_get_connection):
    connection, cursor = _mock_database(("video-123",))
    mock_get_connection.return_value = connection

    assert job_tracker.create_job_queued("video-123") is True

    sql, params = cursor.execute.call_args.args
    assert "ON CONFLICT (video_id) DO NOTHING" in sql
    assert params == ("video-123",)


@patch("app.services.job_tracker.get_connection")
def test_create_job_queued_returns_false_for_duplicate(mock_get_connection):
    connection, _ = _mock_database(None)
    mock_get_connection.return_value = connection

    assert job_tracker.create_job_queued("video-123") is False


@patch("app.services.job_tracker.get_connection")
def test_get_job_status_returns_persisted_record(mock_get_connection):
    connection, _ = _mock_database(("failed", None, "Transcription failed"))
    mock_get_connection.return_value = connection

    result = job_tracker.get_job_status("video-123")

    assert result == {
        "status": "failed",
        "chunks_stored": None,
        "error_message": "Transcription failed",
    }


@patch("app.services.job_tracker.get_connection")
def test_get_job_status_returns_none_for_unknown_video(mock_get_connection):
    connection, _ = _mock_database(None)
    mock_get_connection.return_value = connection

    assert job_tracker.get_job_status("unknown-video") is None


@pytest.mark.parametrize("video_id", ["", "   "])
def test_empty_video_id_is_rejected(video_id):
    with pytest.raises(ValueError, match="video_id cannot be empty"):
        job_tracker.set_job_processing(video_id)
