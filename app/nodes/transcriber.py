# WHAT: Transcribes a local audio file (.wav) into text segments with timestamps
# WHY: We need to convert spoken words from the audio track into readable text for indexing and LLM consumption
# INPUT: local_audio_path (str) - the local path to the WAV audio file on disk (e.g. /tmp/audio_abc.wav)
# OUTPUT: segments (List[Dict[str, Any]]) - a list of dictionaries, each containing 'start', 'end', and 'text'
# WHO CALLS: app/graphs/video_processing_graph.py (as the third node in the pipeline, after audio_extractor)
# CONNECTED TO: faster-whisper AI model, local filesystem /tmp folder

import os
from typing import List, Dict, Any
from faster_whisper import WhisperModel

# Load the faster-whisper model ONCE when this module is first imported,
# not every time transcribe_audio() is called. Loading the model is expensive
# (takes several seconds), so we only want to pay that cost once when the
# FastAPI server starts up, not on every single video processed.
# device="cpu" — change to "cuda" if a GPU is available on the deployment server
# compute_type="int8" — runs efficiently on CPU with lower memory usage
_whisper_model = WhisperModel("base", device="cpu", compute_type="int8")


def transcribe_audio(local_audio_path: str) -> List[Dict[str, Any]]:
    """
    What this does:
        Loads a local WAV file and uses the faster-whisper AI model to transcribe it into text.

    Why it exists:
        To perform Speech-to-Text conversion. We must extract the actual words spoken in the video
        along with timestamps so users can ask questions about specific parts of the video.

    Output:
        Returns a list of dictionaries containing time segments and text.
        Example:
        [
            {"start": 0.0, "end": 2.5, "text": "Hello, welcome to crypto course."},
            {"start": 2.6, "end": 5.8, "text": "Today we will learn about Bitcoin."}
        ]

    Raises:
        ValueError: if the audio file path is empty or None
        FileNotFoundError: if the audio file does not exist on disk
        RuntimeError: if the transcription model fails to execute
    """

    # Step 1: Validate that the input path is not empty
    if not local_audio_path or not local_audio_path.strip():
        raise ValueError("local_audio_path cannot be empty. Please provide a valid audio file path.")

    # Step 2: Check that the audio file actually exists on disk
    if not os.path.exists(local_audio_path):
        raise FileNotFoundError(
            f"Audio file not found at path: {local_audio_path}. "
            "Make sure the audio was extracted successfully before calling this function."
        )

    # Step 3: Check that the audio file is not empty (0 bytes)
    if os.path.getsize(local_audio_path) == 0:
        raise ValueError(f"Audio file at {local_audio_path} is empty (0 bytes). Cannot transcribe empty audio.")

    # Step 4: Run transcription using the already-loaded model
    # beam_size=5 is standard for a good balance of transcription quality and speed.
    try:
        segments_generator, info = _whisper_model.transcribe(local_audio_path, beam_size=5)

        # Step 5: Convert the generator output into a serializable list of dictionaries
        transcribed_segments = []
        for segment in segments_generator:
            transcribed_segments.append({
                "start": round(segment.start, 2),  # Round to 2 decimal places for clean timestamps
                "end": round(segment.end, 2),      # Round to 2 decimal places
                "text": segment.text.strip()       # Remove extra spaces around text
            })

    except Exception as e:
        raise RuntimeError(f"An error occurred during audio transcription: {str(e)}")

    return transcribed_segments