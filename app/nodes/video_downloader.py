# WHAT: Downloads a video file from a remote URL (Cloudinary or S3) to a local temp folder
# WHY: Before FFmpeg can extract audio, we need the video file saved locally on the server
# INPUT: video_url (str) - a full HTTP/HTTPS URL pointing to the video file
# OUTPUT: local_video_path (str) - the local file path where the video is saved (e.g. /tmp/video_abc.mp4)
# WHO CALLS: app/graphs/video_processing_graph.py (as the first node in the pipeline)
# CONNECTED TO: Cloudinary CDN, S3, or any public video URL

import os
import uuid
import requests
import tempfile


def download_video(video_url: str) -> str:
    """
    What this does:
        Takes a video URL and downloads the video file to a temporary folder on the server.

    Why it exists:
        FFmpeg (used in the next step) needs a local file path to extract audio.
        It cannot work directly with remote URLs, so we must download first.

    Output:
        Returns the full local path where the video file is saved.
        Example: /tmp/video_3f8a12bc.mp4

    Raises:
        ValueError: if the video_url is empty or None
        ConnectionError: if the URL is unreachable or returns a non-200 status
        IOError: if the file cannot be written to disk
    """

    # Step 1: Validate that the URL is not empty
    if not video_url or not video_url.strip():
        raise ValueError("video_url cannot be empty. Please provide a valid video URL.")

    # Step 2: Check that the URL starts with http or https
    if not video_url.startswith("http://") and not video_url.startswith("https://"):
        raise ValueError(f"Invalid URL format: '{video_url}'. URL must start with http:// or https://")

    # Step 3: Create a unique temporary file path to save the video
    # We use uuid to generate a unique name so multiple videos don't overwrite each other
    unique_filename = f"video_{uuid.uuid4().hex}.mp4"
    temp_dir = tempfile.gettempdir()  # Gets the system's temp folder (e.g. /tmp on Linux, C:\Users\...\AppData\Local\Temp on Windows)
    local_video_path = os.path.join(temp_dir, unique_filename)

    # Step 4: Try to connect to the URL and download the video
    # Sending a browser-like User-Agent header, because some CDNs/servers block
    # requests that show the default python-requests User-Agent (they assume it's a bot)
    # Using 'with' so the connection is properly closed after download (even if error occurs)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        with requests.get(video_url, stream=True, timeout=60, headers=headers) as response:

            # Step 5: Check if the server responded with success (HTTP 200)
            if response.status_code != 200:
                raise ConnectionError(
                    f"Failed to download video. Server returned status {response.status_code} for URL: {video_url}"
                )

            # Step 6: Write the video content to the local temp file in chunks
            # We write in chunks (8192 bytes at a time) to avoid loading the entire file into memory
            try:
                with open(local_video_path, "wb") as video_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            video_file.write(chunk)
            except IOError as e:
                # If writing fails halfway, delete the partial/broken file from disk
                if os.path.exists(local_video_path):
                    os.remove(local_video_path)
                raise IOError(f"Failed to save video to disk at path: {local_video_path}. Error: {str(e)}")

    except requests.exceptions.ConnectionError:
        raise ConnectionError(f"Could not reach the video URL: {video_url}. Check internet connection or URL validity.")
    except requests.exceptions.Timeout:
        raise ConnectionError(f"Request timed out while trying to download from: {video_url}")

    # Step 7: Confirm the file actually exists and is not empty
    if not os.path.exists(local_video_path) or os.path.getsize(local_video_path) == 0:
        raise IOError(f"Video file was downloaded but appears to be empty or missing at: {local_video_path}")

    return local_video_path