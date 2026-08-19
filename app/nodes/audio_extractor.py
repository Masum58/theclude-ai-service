# WHAT: Extracts audio from a local video file and saves it as a .wav audio file
# WHY: Whisper (speech-to-text model) needs a plain audio file as input, not a video file.
#      FFmpeg handles the conversion from any video format (.mp4, .mkv, etc.) to audio (.wav)
# INPUT: local_video_path (str) - full local path to the downloaded video file (e.g. /tmp/video_abc.mp4)
# OUTPUT: local_audio_path (str) - full local path to the extracted audio file (e.g. /tmp/audio_abc.wav)
# WHO CALLS: app/graphs/video_processing_graph.py (as the second node in the pipeline, after video_downloader)
# CONNECTED TO: FFmpeg (must be installed on the system), local filesystem /tmp folder

import os
import ffmpeg


def extract_audio(local_video_path: str) -> str:
    """
    What this does:
        Takes a local video file path and uses FFmpeg to extract only the audio from it.
        The audio is saved as a .wav file in the same temp folder as the video.

    Why it exists:
        Whisper (our speech-to-text model) only works with audio files, not video files.
        We need to strip out the audio track from the video before we can transcribe it.

    How the audio is extracted:
        - Format: WAV (Waveform Audio File Format) — Whisper works best with this format
        - Sample rate: 16000 Hz (16kHz) — this is the exact rate Whisper expects
        - Channels: 1 (Mono) — single channel audio is enough for transcription

    Output:
        Returns the full local path of the saved .wav audio file.
        Example: /tmp/audio_3f8a12bc.wav

    Raises:
        ValueError: if the video file path is empty or None
        FileNotFoundError: if the video file does not exist at the given path
        RuntimeError: if FFmpeg fails to extract audio (e.g. corrupted video or FFmpeg not installed)
    """

    # Step 1: Validate that the input path is not empty
    if not local_video_path or not local_video_path.strip():
        raise ValueError("local_video_path cannot be empty. Please provide a valid video file path.")

    # Step 2: Check that the video file actually exists on disk
    if not os.path.exists(local_video_path):
        raise FileNotFoundError(
            f"Video file not found at path: {local_video_path}. "
            "Make sure the video was downloaded successfully before calling this function."
        )

    # Step 3: Check that the video file is not empty (0 bytes)
    if os.path.getsize(local_video_path) == 0:
        raise ValueError(f"Video file at {local_video_path} is empty (0 bytes). Cannot extract audio from an empty file.")

    # Step 4: Build the output audio file path
    # We replace the video file extension (.mp4) with .wav to get the audio output path
    # Example: /tmp/video_abc123.mp4 → /tmp/audio_abc123.wav
    base_name = os.path.splitext(os.path.basename(local_video_path))[0]  # e.g. "video_abc123"
    audio_filename = base_name.replace("video_", "audio_") + ".wav"      # e.g. "audio_abc123.wav"
    temp_dir = os.path.dirname(local_video_path)                          # same /tmp folder as the video
    local_audio_path = os.path.join(temp_dir, audio_filename)

    # Step 5: Use FFmpeg to extract audio from the video
    # Settings explained:
    #   acodec='pcm_s16le' → standard WAV audio codec (16-bit PCM), works well with Whisper
    #   ar='16000'         → sample rate of 16000 Hz (16kHz), which is what Whisper expects
    #   ac=1               → mono audio (1 channel), reduces file size and is enough for speech
    try:
        (
            ffmpeg
            .input(local_video_path)
            .output(
                local_audio_path,
                acodec='pcm_s16le',  # 16-bit PCM WAV codec
                ar='16000',          # 16kHz sample rate (Whisper requirement)
                ac=1                 # Mono channel
            )
            .overwrite_output()      # Overwrite if a file with same name already exists
            .run(
                quiet=True,          # Suppress FFmpeg console output (keeps logs clean)
                capture_stderr=True  # Capture error output so we can include it in our error message
            )
        )
    except ffmpeg.Error as e:
        # If FFmpeg fails, clean up any partial audio file that may have been created
        if os.path.exists(local_audio_path):
            os.remove(local_audio_path)

        # Decode the FFmpeg error message for a clear error report
        error_message = e.stderr.decode("utf-8") if e.stderr else "Unknown FFmpeg error"
        raise RuntimeError(
            f"FFmpeg failed to extract audio from: {local_video_path}\n"
            f"FFmpeg error details: {error_message}\n"
            f"Make sure FFmpeg is installed on your system. Install guide: https://ffmpeg.org/download.html"
        )

    # Step 6: Confirm the output audio file was actually created and is not empty
    if not os.path.exists(local_audio_path) or os.path.getsize(local_audio_path) == 0:
        raise RuntimeError(
            f"FFmpeg ran without errors but the audio file is missing or empty at: {local_audio_path}"
        )

    return local_audio_path
