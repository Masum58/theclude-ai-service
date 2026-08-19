# WHAT: Redis worker task for the full video-processing pipeline
# WHY: Downloading, transcribing, embedding, and indexing videos is too heavy to run in
#      the FastAPI web process.
# INPUT: video_id and video_url from an RQ job
# OUTPUT: PostgreSQL job status updated to processing, completed, or failed
# WHO CALLS: an RQ worker consuming the video_processing queue
# CONNECTED TO: video_processing_graph, job_tracker, and temporary-file cleanup

from app.graphs.video_processing_graph import video_processing_app
from app.services.file_cleanup import cleanup_temp_files
from app.services.job_tracker import set_job_completed, set_job_failed, set_job_processing


def run_video_processing(video_id: str, video_url: str) -> None:
    """Execute one queued video pipeline and persist its final outcome."""
    initial_state = {
        "video_id": video_id,
        "video_url": video_url,
        "local_video_path": None,
        "local_audio_path": None,
        "transcript": None,
        "chunks": None,
        "embeddings": None,
        "chunks_stored": None,
        "status": "processing",
        "error_message": None,
    }

    set_job_processing(video_id)

    try:
        final_state = video_processing_app.invoke(initial_state)
    except Exception as error:
        set_job_failed(video_id, f"Unexpected pipeline error: {str(error)}")
        return

    if final_state.get("status") == "completed":
        set_job_completed(video_id, chunks_stored=final_state.get("chunks_stored", 0))
    else:
        set_job_failed(video_id, final_state.get("error_message") or "Unknown processing failure.")

    cleanup_temp_files(
        final_state.get("local_video_path"),
        final_state.get("local_audio_path"),
    )
