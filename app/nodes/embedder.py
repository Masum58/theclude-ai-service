# WHAT: Generates high-dimensional vector embeddings for text chunks using Google Gemini API
# WHY: Converts text into mathematical vectors to enable semantic similarity search in ChromaDB
# INPUT: chunks (List[str]) - the list of text chunks (or a single query string inside a list)
#        task_type (str) - "retrieval_document" for indexing chunks, "retrieval_query" for searching questions
# OUTPUT: embeddings (List[List[float]]) - list of vector embeddings, one vector per input text string
# WHO CALLS: app/graphs/video_processing_graph.py (Phase 3), and Q&A routes (Phase 4)
# CONNECTED TO: Google Generative AI API (Gemini Embedding Service)

from typing import List
import google.generativeai as genai
from app.core.config import settings

# Initialize and configure the Google Generative AI SDK once at module level
# This avoids redundant configure calls during execution
API_KEY = settings.GEMINI_API_KEY
if API_KEY and API_KEY.strip():
    genai.configure(api_key=API_KEY)


def generate_embeddings(chunks: List[str], task_type: str = "retrieval_document") -> List[List[float]]:
    """
    What this does:
        Takes a list of strings and calls Google's Gemini Embedding API to convert them
        into vector embeddings (lists of floats).

    Why it exists:
        To convert text into a numeric representation for similarity search.
        Supports both indexing documents (retrieval_document) and querying (retrieval_query).

    Output:
        Returns a list of float lists (vectors).
        Example: [[0.012, -0.004, ...], [0.087, 0.123, ...]]

    Raises:
        ValueError: if input chunks is empty, or GEMINI_API_KEY is not configured
        RuntimeError: if the Google API call fails
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

    # Step 3: Call Gemini Embedding API in batch mode
    # We use "models/gemini-embedding-001" as text-embedding-004 was deprecated.
    # task_type should be "retrieval_document" for chunks, and "retrieval_query" for user questions.
    try:
        response = genai.embed_content(
            model="models/gemini-embedding-001",
            content=chunks,
            task_type=task_type
        )
    except Exception as e:
        raise RuntimeError(
            f"Google Gemini Embedding API call failed: {str(e)}. "
            "Please check your API key, quota limits, and network connection."
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