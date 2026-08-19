# WHAT: Tests queued-video handling before a user can ask video questions
# WHY: A queued Redis job has no searchable transcript yet and must not be treated as ready.
# INPUT: Mocked persistent job status
# OUTPUT: HTTP 409 while a video waits for a worker

from unittest.mock import patch

import pytest
from fastapi import HTTPException

from app.api.routes.video_qa import _check_video_ready


@patch("app.api.routes.video_qa.get_job_status")
def test_queued_video_cannot_be_queried(mock_get_job_status):
    mock_get_job_status.return_value = {
        "status": "queued",
        "chunks_stored": None,
        "error_message": None,
    }

    with pytest.raises(HTTPException) as exception_info:
        _check_video_ready("video-123")

    assert exception_info.value.status_code == 409
    assert "queued" in exception_info.value.detail
