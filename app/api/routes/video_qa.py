# WHAT: FastAPI routes for asking questions about a processed video (video Q&A) —
#       both a normal (wait-for-full-answer) endpoint and a streaming endpoint
# WHY: Exposes the Q&A pipeline over HTTP so a client can send a question about a video
#      and get back an AI-generated answer, grounded in that video's transcript (via ChromaDB).
# INPUT: HTTP requests — POST /ask-video and POST /ask-video-stream (video_id, question, optional session_id)
# OUTPUT: /ask-video returns QAResponse (JSON). /ask-video-stream returns a plain-text stream,
#         with the session_id in an X-Session-Id response header.
# WHO CALLS: Django backend, or any frontend client showing a chat UI under a video
# CONNECTED TO: app/graphs/qa_graph.py, app/models/chat.py, app/services/job_tracker.py,
#               app/nodes/embedder.py, app/nodes/vector_store.py, app/nodes/llm_answer.py

import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models.chat import QARequest, QAResponse
from app.graphs.qa_graph import qa_app
from app.services.job_tracker import get_job_status
from app.nodes.embedder import generate_embeddings
from app.nodes.vector_store import query_similar_chunks, get_chat_history, save_chat_message, video_has_data
from app.nodes.llm_answer import generate_qa_answer_stream

router = APIRouter()


def _check_video_ready(video_id: str) -> None:
    """
    What this does:
        Confirms a video_id is ready to be queried. First checks the persistent
        PostgreSQL job_tracker. If it has no record — for example, because this
        video was indexed before durable tracking was added — it falls back to checking ChromaDB
        directly for stored chunks before giving up. Raises the appropriate
        HTTPException if the video truly isn't ready. Shared by both /ask-video and
        /ask-video-stream so the same checks aren't duplicated in two places.

    Why the ChromaDB fallback exists:
        Videos indexed before PostgreSQL job tracking was introduced do not have a
        corresponding job record, even though their chunks and embeddings remain
        safely persisted in ChromaDB. The fallback keeps those existing videos queryable.

    Raises:
        HTTPException 404: if this video_id has no job record AND no data in ChromaDB
        HTTPException 409: if the video is queued or still processing
        HTTPException 500: if job_tracker says the video's processing failed
    """
    job = get_job_status(video_id)

    if job is None:
        # No in-memory record — check ChromaDB before concluding this video was
        # never processed. It might just be that the server restarted since then.
        if video_has_data(video_id):
            return  # Data exists — treat this video as ready, even without a job record
        raise HTTPException(
            status_code=404,
            detail=f"video_id '{video_id}' was never submitted for processing. "
                   f"Call POST /upload first."
        )

    if job["status"] in ("queued", "processing"):
        status_message = (
            "is queued and waiting for a processing worker"
            if job["status"] == "queued"
            else "is still processing"
        )
        raise HTTPException(
            status_code=409,
            detail=f"video_id '{video_id}' {status_message}. "
            f"Try again shortly, or poll GET /status/{video_id}."
        )

    if job["status"] == "failed":
        raise HTTPException(
            status_code=500,
            detail=f"video_id '{video_id}' failed during processing, so it has no "
                   f"searchable content. Original error: {job['error_message']}"
        )

    # job["status"] == "completed" — nothing more to check, video is ready


@router.post("/ask-video", response_model=QAResponse)
async def ask_video_question(request: QARequest):
    """
    What this does:
        Runs the qa_graph pipeline (embed question + fetch history in parallel, query
        ChromaDB, generate an answer with Gemini, save the turn to chat history) and
        returns the complete answer in one response, once it's fully generated.

    Why it exists:
        This is the HTTP entry point for the "ask the video a question" feature —
        everything qa_graph.py does internally becomes reachable over the network here.
        For a streamed, progressively-arriving version of the same feature, see
        /ask-video-stream below.
    """
    _check_video_ready(request.video_id)

    session_id = request.session_id or f"session_{uuid.uuid4().hex[:12]}"

    initial_state = {
        "video_id": request.video_id,
        "session_id": session_id,
        "question": request.question,
        "query_embedding": None,
        "chat_history": None,
        "context": None,
        "answer": None,
        "error_message": None,
    }

    final_state = qa_app.invoke(initial_state)

    return QAResponse(
        session_id=session_id,
        answer=final_state.get("answer"),
        error_message=final_state.get("error_message"),
    )


@router.post("/ask-video-stream")
async def ask_video_question_stream(request: QARequest):
    """
    What this does:
        Same overall flow as /ask-video (embed question, fetch history, query ChromaDB,
        generate an answer, save the turn to chat history) — but the answer is streamed
        back to the client piece by piece as Gemini generates it, instead of waiting for
        the complete answer before responding.

    Why it exists:
        /ask-video makes the client wait for the entire answer before seeing anything.
        This endpoint sends the first words back almost immediately and streams the rest,
        which feels much more responsive for a chat-style UI, even though the total time
        to finish generating is about the same.

    Why this doesn't reuse qa_app (the LangGraph pipeline):
        LangGraph nodes return a finished value, not a generator — mixing streaming output
        into that graph would need a different node design. It's simpler and clearer to
        call the same underlying functions directly here, in a straight sequence, and only
        stream the one step that actually benefits from it (the LLM call).

    Why session_id is returned in a header instead of the response body:
        The response body here is a raw stream of answer text, not JSON — there's no
        natural place inside it to also include the session_id. The X-Session-Id header
        lets the client read it without it interfering with the streamed text.

    Raises:
        HTTPException 404/409/500: same job-status checks as /ask-video, for the same reasons.
    """
    _check_video_ready(request.video_id)

    session_id = request.session_id or f"session_{uuid.uuid4().hex[:12]}"

    query_embedding = generate_embeddings([request.question], task_type="retrieval_query")[0]
    history = get_chat_history(session_id=session_id, video_id=request.video_id)
    context = query_similar_chunks(query_embedding=query_embedding, video_id=request.video_id, top_k=5)

    def stream_and_save():
        full_answer_parts = []
        for chunk in generate_qa_answer_stream(question=request.question, context=context, history=history):
            full_answer_parts.append(chunk)
            yield chunk

        full_answer = "".join(full_answer_parts)
        save_chat_message(session_id=session_id, video_id=request.video_id, role="user", message=request.question)
        save_chat_message(session_id=session_id, video_id=request.video_id, role="assistant", message=full_answer)

    return StreamingResponse(
        stream_and_save(),
        media_type="text/plain",
        headers={"X-Session-Id": session_id}
    )
