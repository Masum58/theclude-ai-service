# WHAT: FastAPI routes for triggering video processing and checking its status
# WHY: Accepts a video URL, queues its processing in Redis/RQ, and lets Django (or any
#      client) poll persistent PostgreSQL status without blocking on the pipeline.
# INPUT: HTTP requests — POST /upload (video_id, video_url), GET /status/{video_id}
# OUTPUT: HTTP JSON responses matching VideoProcessResponse / JobStatusResponse
# WHO CALLS: Django backend (or Postman/curl for manual testing)
# CONNECTED TO: app.services.video_queue, app.services.job_tracker.py, app.models.video.py

from fastapi import APIRouter, HTTPException

from app.models.video import VideoProcessRequest, VideoProcessResponse, JobStatusResponse
from app.services.job_tracker import (
    set_job_failed,
    create_job_queued,
    get_job_status,
)
from app.services.video_queue import enqueue_video_processing

router = APIRouter()


@router.post("/upload", response_model=VideoProcessResponse)
async def upload_video(request: VideoProcessRequest):
    """
    What this does:
        Accepts a video_id + video_url, atomically creates a "queued" PostgreSQL job,
        sends the pipeline to Redis/RQ, and returns immediately.

    Why it exists:
        This is the fix for the original latency problem: the old Django UploadVideo view
        blocked the client until the entire pipeline finished. Here, the HTTP response comes
        back in milliseconds, and the client polls GET /status/{video_id} to know when it's done.
    """
    if not create_job_queued(request.video_id):
        existing_job = get_job_status(request.video_id)
        raise HTTPException(
            status_code=409,
            detail=(
                f"video_id '{request.video_id}' has already been submitted "
                f"with status '{existing_job['status']}'."
            ),
        )

    try:
        enqueue_video_processing(request.video_id, request.video_url)
    except Exception as error:
        set_job_failed(request.video_id, f"Failed to queue video processing: {str(error)}")
        raise HTTPException(status_code=503, detail="Video processing queue is unavailable.")

    return VideoProcessResponse(
        video_id=request.video_id,
        status="queued",
        message="Video processing queued successfully."
    )


@router.get("/status/{video_id}", response_model=JobStatusResponse)
async def get_video_status(video_id: str):
    """
    What this does:
        Looks up the current processing status for a given video_id and returns it.

    Why it exists:
        Lets the client (Django backend) poll to find out when processing has finished,
        instead of blocking on the /upload call.

    Raises:
        HTTPException 404: if this video_id has never been submitted for processing.
    """
    job = get_job_status(video_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"No job found for video_id '{video_id}'. It may not have been submitted, "
                   f"or the server restarted since it was."
        )

    return JobStatusResponse(
        video_id=video_id,
        status=job["status"],
        chunks_stored=job["chunks_stored"],
        error_message=job["error_message"],
    )
