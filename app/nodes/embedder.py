# WHAT: Generates high-dimensional vector embeddings for text chunks using Google Gemini API
# WHY: Converts text into mathematical vectors to enable semantic similarity search in ChromaDB
# INPUT: chunks (List[str]) - the list of text chunks (or a single query string inside a list)
#        task_type (str) - "retrieval_document" for indexing chunks, "retrieval_query" for searching questions
# OUTPUT: embeddings (List[List[float]]) - list of vector embeddings, one vector per input text string
# WHO CALLS: app/graphs/video_processing_graph.py (Phase 3), and Q&A routes (Phase 4)
# CONNECTED TO: Google Generative AI API (Gemini Embedding Service)

import time
from typing import List
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from app.core.config import settings

# Initialize and configure the Google Generative AI SDK once at module level
# This avoids redundant configure calls during execution
API_KEY = settings.GEMINI_API_KEY
if API_KEY and API_KEY.strip():
    genai.configure(api_key=API_KEY)

# Retry settings for rate-limit (429) errors. Free-tier Gemini quotas are easy to hit
# when processing many videos back-to-back (e.g. a Cloudinary backfill of 200+ videos),
# so instead of failing the whole pipeline on the first 429, we wait and retry a few
# times with increasing delays before giving up.
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 15


def generate_embeddings(chunks: List[str], task_type: str = "retrieval_document") -> List[List[float]]:
    """
    What this does:
        Takes a list of strings and calls Google's Gemini Embedding API to convert them
        into vector embeddings (lists of floats). Automatically retries with increasing
        delays if the API responds with a rate-limit (429) error.

    Why it exists:
        To convert text into a numeric representation for similarity search.
        Supports both indexing documents (retrieval_document) and querying (retrieval_query).

    Why retry logic was added:
        Gemini's free tier allows only a handful of requests per minute. When processing
        many videos in a row (e.g. backfilling an existing Cloudinary library), hitting
        that limit is expected, not exceptional — so a single 429 shouldn't fail the
        entire video-processing job. Retrying with backoff lets the pipeline pause and
        continue automatically instead of the caller needing to handle this itself.

    Output:
        Returns a list of float lists (vectors).
        Example: [[0.012, -0.004, ...], [0.087, 0.123, ...]]

    Raises:
        ValueError: if input chunks is empty, or GEMINI_API_KEY is not configured
        RuntimeError: if the Google API call fails for a non-rate-limit reason, or if
                      rate-limit retries are exhausted
    """

    # Step 1: Validate input arguments
    if not isinstance(chunks, list):
        raise ValueError("Input 'chunks' must be a list of strings.")

    # If the chunks list is empty, return an empty list immediately
    if not chunks:
        return []

    # Step 2: Ensure API key was configured successfully
    if not API_KEY or not API_KEY.strip():
        raise ValueError(
            "GEMINI_API_KEY is missing or empty in configuration (.env). "
            "Please configure your Google Gemini API key to proceed."
        )

    # Step 3: Call Gemini Embedding API in batch mode, retrying on rate-limit errors.
    # We use "models/gemini-embedding-001" as text-embedding-004 was deprecated.
    # task_type should be "retrieval_document" for chunks, and "retrieval_query" for user questions.
    response = None
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = genai.embed_content(
                model="models/gemini-embedding-001",
                content=chunks,
                task_type=task_type
            )
            break  # success — stop retrying
        except ResourceExhausted as e:
            # This is Google's specific exception for 429 rate-limit responses.
            last_error = e
            if attempt == MAX_RETRIES:
                break  # out of retries, fall through to the error below
            wait_seconds = INITIAL_BACKOFF_SECONDS * attempt  # 15s, 30s, 45s, 60s...
            print(f"    ⏳ Embedding rate limit (429) hit. Waiting {wait_seconds}s before retry ({attempt}/{MAX_RETRIES})...")
            time.sleep(wait_seconds)
        except Exception as e:
            # Any other kind of failure (bad key, network issue, etc.) is not something
            # retrying will fix, so fail immediately instead of wasting time retrying.
            raise RuntimeError(
                f"Google Gemini Embedding API call failed: {str(e)}. "
                "Please check your API key, quota limits, and network connection."
            )

    if response is None:
        raise RuntimeError(
            f"Google Gemini Embedding API call failed after {MAX_RETRIES} rate-limit retries: {str(last_error)}. "
            "Consider upgrading to a paid Gemini API tier if this happens often."
        )

    # Step 4: Extract and validate embeddings from response
    embeddings = response.get("embedding", [])

    if not embeddings:
        raise RuntimeError("Received an empty response from Google Gemini Embedding API.")

    # Confirm that we received exactly one vector for each input text string
    if len(embeddings) != len(chunks):
        raise RuntimeError(
            f"Embedding mismatch: expected {len(chunks)} vectors but got {len(embeddings)} from Google API."
        )

    return embeddings