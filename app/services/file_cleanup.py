# WHAT: Deletes temporary video/audio files from disk after processing is done
# WHY: Once a video's chunks and embeddings are persisted in ChromaDB, the raw downloaded
#      video (.mp4) and extracted audio (.wav) files serve no further purpose. Without
#      cleanup, every /upload call leaves two files behind permanently, and the temp
#      folder grows without bound over time.
# INPUT: local_video_path (str, optional), local_audio_path (str, optional)
# OUTPUT: None — deletes files as a side effect, never raises on missing/already-deleted files
# WHO CALLS: app/api/routes/video.py (run_video_processing, after the graph finishes)
# CONNECTED TO: Local filesystem only

import os
from typing import Optional


def cleanup_temp_files(local_video_path: Optional[str], local_audio_path: Optional[str]) -> None:
    """
    What this does:
        Deletes the given video and/or audio file paths from disk, if they exist.

    Why it exists:
        Called once at the end of video processing (success or failure) so temp files
        don't accumulate. Takes Optional paths because a failed pipeline might have only
        gotten as far as downloading the video (no audio file yet), or might have failed
        before downloading anything at all (both paths still None).

    Design note:
        This function never raises. A missing file, a permission error, or a None path
        are all just skipped silently (with a print for visibility) — cleanup failing
        should never crash the request that's already been handled, since by this point
        the job's real status (completed/failed) has already been recorded.

    Output:
        None. Prints a short confirmation for each file it actually deletes, for visibility
        during development. (Can be swapped for proper logging later.)
    """
    for label, path in [("video", local_video_path), ("audio", local_audio_path)]:
        if not path:
            # Nothing was ever created at this stage (e.g. pipeline failed before this step)
            continue

        if not os.path.exists(path):
            # Already deleted, or path was never valid — nothing to do
            continue

        try:
            os.remove(path)
            print(f"[cleanup] Deleted temp {label} file: {path}")
        except OSError as e:
            # Don't let a cleanup failure affect the already-completed request —
            # just note it so it's visible if someone's watching the server logs.
            print(f"[cleanup] WARNING: could not delete {label} file at {path}: {e}")