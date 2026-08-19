# WHAT: Defines the LangGraph pipeline for full video processing (download → audio → transcribe → chunk → embed → store)
# WHY: LangGraph allows us to connect individual node functions into a controlled, stateful workflow.
#      Each node reads from and writes to a shared GraphState, making data flow predictable and debuggable.
# INPUT: Initial state with video_id and video_url
# OUTPUT: Final state with transcript, chunks, embeddings, and status (completed/failed).
#         On success, chunks + embeddings are also persisted in ChromaDB (not just in the state).
# WHO CALLS: RQ worker task in app/tasks/video_processing.py
# CONNECTED TO: All nodes in app/nodes/, app/nodes/vector_store.py (ChromaDB), app/services/job_tracker.py

from typing import TypedDict, List, Any, Optional
from langgraph.graph import StateGraph, END

from app.nodes.video_downloader import download_video
from app.nodes.audio_extractor import extract_audio
from app.nodes.transcriber import transcribe_audio
from app.nodes.text_chunker import chunk_transcript
from app.nodes.embedder import generate_embeddings
from app.nodes.vector_store import store_video_chunks

from app.services.job_tracker import (
    set_job_processing,
    set_job_completed,
    set_job_failed,
)

# ─────────────────────────────────────────────
# GRAPH STATE DEFINITION
# ─────────────────────────────────────────────

class GraphState(TypedDict):
    """
    What this is:
        The central data container that is passed between every node in the pipeline.
        Think of it as a relay baton — each node picks it up, adds its results, and passes it on.

    Fields:
        video_id          : Unique identifier for the video (UUID from Django database)
        video_url         : Remote URL (Cloudinary/S3) where the video file is hosted
        local_video_path  : Local /tmp path where the video file is downloaded
        local_audio_path  : Local /tmp path where the extracted audio (.wav) is saved
        transcript        : List of dicts with start, end, text from Whisper transcription
        chunks            : List of formatted strings (timestamped text chunks for indexing)
        embeddings        : List of float vectors (one per chunk, from Gemini Embedding API)
        chunks_stored     : Number of chunks successfully persisted in ChromaDB
        status            : Current pipeline status ("processing", "completed", "failed")
        error_message     : If any node fails, the error description is stored here
    """
    video_id: str
    video_url: str
    local_video_path: Optional[str]
    local_audio_path: Optional[str]
    transcript: Optional[List[Any]]
    chunks: Optional[List[str]]
    embeddings: Optional[List[List[float]]]
    chunks_stored: Optional[int]
    status: str
    error_message: Optional[str]


# ─────────────────────────────────────────────
# NODE WRAPPER FUNCTIONS
# ─────────────────────────────────────────────

def node_download_video(state: GraphState) -> GraphState:
    """
    What this does:
        Calls the download_video() function to download the remote video to /tmp.
        Also marks the job as "processing" in PostgreSQL right as work begins.
    """
    try:
        set_job_processing(state["video_id"])

        local_path = download_video(state["video_url"])

        return {
            **state,
            "local_video_path": local_path,
            "status": "processing",
        }

    except Exception as e:
        error_message = f"[node_download_video] {str(e)}"

        set_job_failed(
            video_id=state["video_id"],
            error_message=error_message,
        )

        return {
            **state,
            "status": "failed",
            "error_message": error_message,
        }


def node_extract_audio(state: GraphState) -> GraphState:
    """
    What this does:
        Calls the extract_audio() function to extract .wav audio from the downloaded video.
        This node only runs if the download step succeeded (enforced by routing).
    """
    try:
        audio_path = extract_audio(state["local_video_path"])
        return {**state, "local_audio_path": audio_path}
    except Exception as e:
        error_message = f"[node_extract_audio] {str(e)}"

        set_job_failed(
            video_id=state["video_id"],
            error_message=error_message,
        )

        return {
            **state,
            "status": "failed",
            "error_message": error_message,
        }


def node_transcribe(state: GraphState) -> GraphState:
    """
    What this does:
        Calls the transcribe_audio() function to convert audio into timestamped text segments.
    """
    try:
        segments = transcribe_audio(state["local_audio_path"])
        return {**state, "transcript": segments}
    except Exception as e:
        error_message = f"[node_transcribe] {str(e)}"
        set_job_failed(
            video_id=state["video_id"],
            error_message=error_message,
        )

        return {
            **state,
            "status": "failed",
            "error_message": error_message,
        }


def node_chunk(state: GraphState) -> GraphState:
    """
    What this does:
        Calls the chunk_transcript() function to group transcript segments into sized text chunks.
    """
    try:
        chunks = chunk_transcript(state["transcript"])
        return {**state, "chunks": chunks}
    except Exception as e:
        error_message = f"[node_chunk] {str(e)}"

        set_job_failed(
            video_id=state["video_id"],
            error_message=error_message,
        )
        return {
            **state,
            "status": "failed",
            "error_message": error_message,
        }


def node_embed(state: GraphState) -> GraphState:
    """
    What this does:
        Calls the generate_embeddings() function to convert all text chunks into vector embeddings.
        This is no longer the last node — store_chromadb runs after this to persist the result.
    """
    try:
        embeddings = generate_embeddings(state["chunks"], task_type="retrieval_document")
        return {**state, "embeddings": embeddings}
    except Exception as e:
        error_message = f"[node_embed] {str(e)}"

        set_job_failed(
            video_id=state["video_id"],
            error_message=error_message,
        )
        return {
            **state,
            "status": "failed",
            "error_message": error_message,
        }


def node_store_chromadb(state: GraphState) -> GraphState:
    """
    What this does:
        Calls store_video_chunks() to persist this video's chunks + embeddings in ChromaDB,
        tagged with video_id. This is the final step — on success, the pipeline status
        becomes "completed".

    Why it exists:
        Everything before this node only existed in memory (the graph state). Without this
        step, all that work would be lost as soon as the process finished — nothing would
        be searchable later when a user asks a question about this video.
    """
    try:
        stored_count = store_video_chunks(
            video_id=state["video_id"],
            chunks=state["chunks"],
            embeddings=state["embeddings"]
        )
        set_job_completed(
            video_id=state["video_id"],
            chunks_stored=stored_count,
        )

        return {
            **state,
            "chunks_stored": stored_count,
            "status": "completed",
            "error_message": None,
        }
    except Exception as e:
        error_message = f"[node_store_chromadb] {str(e)}"

        set_job_failed(
            video_id=state["video_id"],
            error_message=error_message,
        )
        return {
            **state,
            "status": "failed",
            "error_message": error_message,
        }


# ─────────────────────────────────────────────
# ROUTING FUNCTION
# ─────────────────────────────────────────────

def check_for_error(state: GraphState) -> str:
    """
    What this does:
        Inspects the state after a node runs and decides the next step.
        Returns "error" if a failure was recorded (routes to END),
        or "continue" to proceed to the next node in the pipeline.
    """
    if state.get("error_message"):
        return "error"
    return "continue"


# ─────────────────────────────────────────────
# GRAPH ASSEMBLY
# ─────────────────────────────────────────────

def build_video_processing_graph():
    """
    What this does:
        Assembles all 6 node wrapper functions into a sequential LangGraph pipeline.
        The graph flows: download → extract_audio → transcribe → chunk → embed →
        store_chromadb → END
    """
    graph = StateGraph(GraphState)

    graph.add_node("download_video", node_download_video)
    graph.add_node("extract_audio", node_extract_audio)
    graph.add_node("transcribe", node_transcribe)
    graph.add_node("chunk", node_chunk)
    graph.add_node("embed", node_embed)
    graph.add_node("store_chromadb", node_store_chromadb)

    graph.set_entry_point("download_video")

    graph.add_conditional_edges(
        "download_video",
        check_for_error,
        {"continue": "extract_audio", "error": END}
    )
    graph.add_conditional_edges(
        "extract_audio",
        check_for_error,
        {"continue": "transcribe", "error": END}
    )
    graph.add_conditional_edges(
        "transcribe",
        check_for_error,
        {"continue": "chunk", "error": END}
    )
    graph.add_conditional_edges(
        "chunk",
        check_for_error,
        {"continue": "embed", "error": END}
    )
    graph.add_conditional_edges(
        "embed",
        check_for_error,
        {"continue": "store_chromadb", "error": END}
    )
    graph.add_edge("store_chromadb", END)

    return graph.compile()


# Build the graph once at module level so it's ready to be imported and invoked
video_processing_app = build_video_processing_graph()