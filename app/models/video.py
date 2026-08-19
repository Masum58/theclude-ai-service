# WHAT: Pydantic request/response schemas for the video processing and status-check API endpoints
# WHY: FastAPI uses these schemas to automatically validate incoming requests and document the API
#      (visible in /docs). This keeps validation logic out of the route functions themselves.
# INPUT: N/A (this file only defines shapes of data, not logic)
# OUTPUT: N/A
# WHO CALLS: app/api/routes/video.py (Phase 4)
# CONNECTED TO: app/graphs/video_processing_graph.py (video_id is the same one used throughout the pipeline)

from pydantic import BaseModel, Field
from typing import Optional


class VideoProcessRequest(BaseModel):
    """
    What this is:
        The expected request body when Django (or any client) asks us to process a video.

    Why it exists:
        FastAPI validates incoming JSON against this schema automatically — if video_id or
        video_url is missing or the wrong type, the request is rejected before our route
        code even runs, with a clear error message.

    Fields:
        video_id  : Unique identifier for the video. We reuse this single ID as the "job ID"
                    too, since one video_id always maps to exactly one processing job —
                    there's no need for a separate job_id.
        video_url : Full HTTP(S) URL where the video file can be downloaded (Cloudinary/S3).
    """
    video_id: str = Field(..., min_length=1, description="Unique ID of the video (from Django)")
    video_url: str = Field(..., min_length=1, description="Public URL where the video file can be downloaded")

    class Config:
        json_schema_extra = {
            "example": {
                "video_id": "abc123",
                "video_url": "https://res.cloudinary.com/example/video/upload/v1/lesson1.mp4"
            }
        }


class VideoProcessResponse(BaseModel):
    """
    What this is:
        The response sent back immediately after a /upload request is accepted — before
        processing has actually finished. This confirms the job started and tells the
        client how to check on it later.

    Fields:
        video_id : Echoes back the same ID the client sent — this doubles as the job ID
                   to use with GET /status/{video_id}.
        status   : "queued" at this point, until an RQ worker starts processing it.
        message  : A short human-readable confirmation message.
    """
    video_id: str
    status: str = "queued"
    message: str = "Video processing queued successfully."


class JobStatusResponse(BaseModel):
    """
    What this is:
        The response shape for GET /status/{video_id} — lets a client poll to see how
        a previously-submitted video is progressing.

    Fields:
        video_id      : The ID being checked.
        status        : One of "queued", "processing", "completed", or "failed".
        chunks_stored : How many chunks were saved to ChromaDB (only meaningful once
                        status is "completed"). None while still processing or if failed.
        error_message : Populated only when status is "failed" — explains what went wrong.
    """
    video_id: str
    status: str
    chunks_stored: Optional[int] = None
    error_message: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "video_id": "abc123",
                "status": "completed",
                "chunks_stored": 8,
                "error_message": None
            }
        }
