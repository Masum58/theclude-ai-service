# WHAT: Provides all ChromaDB operations — storing video chunks, semantic search, and chat history management
# WHY: ChromaDB is our vector database. It stores text chunks as mathematical vectors so we can find
#      the most relevant parts of a video transcript when a user asks a question (semantic search).
# INPUT: video_id, chunks, embeddings, questions, session_id, chat messages
# OUTPUT: stored confirmations, list of relevant text chunks, chat history strings
# WHO CALLS: app/graphs/video_processing_graph.py (store), app/graphs/qa_graph.py (query + history)
# CONNECTED TO: ChromaDB (local persistent storage in chroma_db/ folder), app/core/config.py, app/nodes/embedder.py

import uuid
from typing import List, Dict, Any, Optional
import chromadb
from app.core.config import settings
from app.nodes.embedder import generate_embeddings


# ─────────────────────────────────────────────
# CHROMADB CLIENT & COLLECTIONS (initialized once at module level)
# ─────────────────────────────────────────────

# Create a persistent ChromaDB client — data is saved to disk and survives server restarts
# The chroma_db/ folder is defined in our .env via CHROMA_DB_DIR setting
chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DB_DIR)

# Collection 1: Stores video transcript chunks + their embeddings (one entry per chunk)
video_chunks_collection = chroma_client.get_or_create_collection(
    name="video_knowledge_base"
)

# Collection 2: Stores chat messages (user questions + assistant answers) per session
chat_history_collection = chroma_client.get_or_create_collection(
    name="chat_history"
)


# ─────────────────────────────────────────────
# STORE FUNCTIONS
# ─────────────────────────────────────────────

def store_video_chunks(
    video_id: str,
    chunks: List[str],
    embeddings: List[List[float]]
) -> int:
    """
    What this does:
        Saves all the text chunks from a video transcript (along with their vector embeddings)
        into ChromaDB. Each chunk is tagged with the video_id so we can filter by video later.

    Why it exists:
        After Whisper transcribes and we chunk the text, we need to persist these chunks
        in a vector database so that during Q&A we can retrieve only the relevant parts.

    Note:
        Unlike save_chat_message(), this function still expects pre-computed embeddings
        from the caller (the video_processing_graph), because embedding generation there
        is already batched as part of the pipeline's embed step — recomputing it here
        would mean calling the Gemini API twice for the same chunks.

    Output:
        Returns the number of chunks successfully stored.

    Raises:
        ValueError: if inputs are empty or mismatched in length
        RuntimeError: if ChromaDB storage fails
    """

    # Step 1: Validate inputs
    if not video_id or not video_id.strip():
        raise ValueError("video_id cannot be empty.")
    if not chunks:
        raise ValueError("chunks list cannot be empty.")
    if not embeddings:
        raise ValueError("embeddings list cannot be empty.")
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings. They must be equal."
        )

    # Step 2: Build unique IDs for each chunk
    # Each chunk needs a unique ID in ChromaDB — we combine video_id with an index
    chunk_ids = [f"{video_id}_chunk_{i}" for i in range(len(chunks))]

    # Step 3: Build metadata for each chunk so we can filter by video_id during search
    metadatas = [{"video_id": video_id} for _ in chunks]

    # Step 4: Store everything in ChromaDB in a single batch call
    try:
        video_chunks_collection.upsert(
            ids=chunk_ids,           # Unique IDs per chunk
            documents=chunks,        # The actual text content
            embeddings=embeddings,   # Pre-computed float vectors
            metadatas=metadatas      # Filter metadata (video_id)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to store chunks in ChromaDB: {str(e)}")

    return len(chunks)


def save_chat_message(
    session_id: str,
    video_id: str,
    role: str,
    message: str
) -> None:
    """
    What this does:
        Saves a single chat message (either from the user or the assistant) into the
        chat history ChromaDB collection, tagged with session_id and video_id.
        The embedding for this message is generated internally — the caller only needs
        to pass the plain text.

    Why it exists:
        To maintain conversational context. When a user asks a follow-up question,
        we can retrieve the previous conversation turns and include them in the LLM prompt.

    Why the embedding is generated here (not by the caller):
        Callers of this function (like qa_graph.py) shouldn't need to remember to embed
        the message themselves before calling this — that's an easy step to forget and a
        detail that belongs to storage, not to the graph logic. Keeping it here means
        "save this message" is the only thing a caller needs to think about.

    Output:
        No return value. Raises an error if embedding or saving fails.

    Raises:
        ValueError: if any required input is empty or role is invalid
        RuntimeError: if embedding generation or ChromaDB storage fails
    """

    # Step 1: Validate inputs
    if not session_id or not video_id or not message or not role:
        raise ValueError("session_id, video_id, role, and message are all required.")
    if role not in ("user", "assistant"):
        raise ValueError(f"role must be 'user' or 'assistant', got: '{role}'")

    # Step 2: Generate the embedding for this message right here.
    # generate_embeddings() expects a list, so we wrap the single message in one,
    # then take the first (and only) vector back out.
    try:
        embedding = generate_embeddings([message], task_type="retrieval_document")[0]
    except Exception as e:
        raise RuntimeError(f"Failed to generate embedding for chat message: {str(e)}")

    # Step 3: Save to ChromaDB with a unique ID and metadata for filtering
    try:
        chat_history_collection.upsert(
            ids=[str(uuid.uuid4())],
            documents=[message],
            embeddings=[embedding],
            metadatas=[{
                "session_id": session_id,
                "video_id": video_id,
                "role": role
            }]
        )
    except Exception as e:
        raise RuntimeError(f"Failed to save chat message to ChromaDB: {str(e)}")


# ─────────────────────────────────────────────
# QUERY FUNCTIONS
# ─────────────────────────────────────────────

def query_similar_chunks(
    query_embedding: List[float],
    video_id: str,
    top_k: int = 5
) -> str:
    """
    What this does:
        Searches ChromaDB for the most semantically similar text chunks to the user's question,
        filtered to only the chunks belonging to the specified video.

    Why it exists:
        Instead of sending the full transcript to the LLM (expensive, slow), we find only
        the 3-5 most relevant chunks and send those as context. This is the RAG pattern.

    Output:
        Returns a single string of all retrieved chunks joined together (ready for LLM prompt).
        Example: "[00:00:10 - 00:01:30] Bitcoin is a decentralized... [00:03:00 - 00:04:00] Mining is..."

    Raises:
        ValueError: if query_embedding or video_id is empty
        RuntimeError: if ChromaDB query fails
    """

    # Step 1: Validate inputs
    if not query_embedding:
        raise ValueError("query_embedding cannot be empty.")
    if not video_id or not video_id.strip():
        raise ValueError("video_id cannot be empty.")

    # Step 2: Query ChromaDB for the top_k closest chunks for this specific video
    # The 'where' filter ensures we only get chunks from the requested video
    try:
        results = video_chunks_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"video_id": video_id}   # DB-level filter — no Python-side filtering needed
        )
    except Exception as e:
        raise RuntimeError(f"ChromaDB semantic search failed: {str(e)}")

    # Step 3: Extract and join the matched documents into a single context string
    matched_docs = results.get("documents", [[]])[0]
    if not matched_docs:
        return ""

    return "\n\n".join(matched_docs)


def get_chat_history(
    session_id: str,
    video_id: str,
    last_n: int = 6
) -> str:
    """
    What this does:
        Retrieves the last N messages from the chat history for a given session and video,
        formatted as a readable conversation string for the LLM prompt.

    Why it exists:
        The LLM needs context of the previous conversation turns to give coherent follow-up answers.
        Without history, every answer would treat each question as if it were the first.

    Output:
        Returns a formatted string of the conversation history.
        Example:
            User: What is Bitcoin?
            Assistant: Bitcoin is a decentralized digital currency...

    Raises:
        ValueError: if session_id or video_id is empty
        RuntimeError: if ChromaDB query fails
    """

    # Step 1: Validate inputs
    if not session_id or not video_id:
        raise ValueError("session_id and video_id are both required.")

    # Step 2: Fetch messages from ChromaDB filtered by both session_id and video_id at the DB level
    try:
        results = chat_history_collection.get(
            where={
                "$and": [
                    {"session_id": {"$eq": session_id}},
                    {"video_id":   {"$eq": video_id}}
                ]
            }
        )
    except Exception as e:
        raise RuntimeError(f"Failed to fetch chat history from ChromaDB: {str(e)}")

    # Step 3: If no history exists yet, return an empty string
    docs = results.get("documents", [])
    metas = results.get("metadatas", [])
    if not docs:
        return ""

    # Step 4: Format messages as "Role: Message" lines and return the last N messages
    formatted_messages = []
    for doc, meta in zip(docs, metas):
        role = meta.get("role", "user").capitalize()
        formatted_messages.append(f"{role}: {doc.strip()}")

    # Return only the last N messages to keep the context window manageable
    return "\n".join(formatted_messages[-last_n:])
    

def video_has_data(video_id: str) -> bool:
    """
    What this does:
        Checks whether ChromaDB has at least one stored chunk for the given video_id,
        without needing a query embedding — just a metadata lookup.

    Why it exists:
        job_tracker is in-memory only, so its records are lost whenever the server
        restarts. But the actual chunk data in ChromaDB survives restarts (it's written
        to disk). This function lets the Q&A route fall back to checking ChromaDB
        directly when job_tracker has no record for a video_id — so a previously
        processed video doesn't incorrectly appear as "never uploaded" just because
        the server happened to restart since then.

    Output:
        True if at least one chunk exists for this video_id, False otherwise.

    Raises:
        ValueError: if video_id is empty
        RuntimeError: if the ChromaDB lookup fails
    """
    if not video_id or not video_id.strip():
        raise ValueError("video_id cannot be empty.")

    try:
        # limit=1 is enough — we only need to know if anything exists at all,
        # not how many chunks there are
        results = video_chunks_collection.get(
            where={"video_id": video_id},
            limit=1
        )
    except Exception as e:
        raise RuntimeError(f"Failed to check ChromaDB for video_id '{video_id}': {str(e)}")

    docs = results.get("documents", [])
    return len(docs) > 0