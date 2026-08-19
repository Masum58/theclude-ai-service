# WHAT: Defines the LangGraph pipeline for video Q&A with conversational memory (parallel embedding & history fetch)
# WHY: LangGraph enables parallel node execution (fan-out) for independent tasks like query embedding
#      and history fetching. It then merges the state (fan-in) to query ChromaDB and generate an answer using LLM.
# INPUT: Initial state with video_id, session_id, and question
# OUTPUT: Final state containing context, history, and generated answer
# WHO CALLS: FastAPI endpoint in api/routes/video_qa.py (Phase 4)
# CONNECTED TO: app/nodes/embedder.py, app/nodes/vector_store.py, app/nodes/llm_answer.py

from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END

from app.nodes.embedder import generate_embeddings
from app.nodes.vector_store import query_similar_chunks, get_chat_history, save_chat_message
from app.nodes.llm_answer import generate_qa_answer


# ─────────────────────────────────────────────
# GRAPH STATE DEFINITION
# ─────────────────────────────────────────────

class QAState(TypedDict):
    """
    What this is:
        The state container for the video Q&A pipeline.
        Manages inputs (question/session) and intermediate outputs (embeddings/context/history/answer).
    """
    video_id: str
    session_id: str
    question: str
    query_embedding: Optional[List[float]]
    chat_history: Optional[str]
    context: Optional[str]
    answer: Optional[str]
    error_message: Optional[str]


# ─────────────────────────────────────────────
# NODE WRAPPER FUNCTIONS
# IMPORTANT: Every node below returns only the fields it changed (a partial dict),
# never the full state via {**state, ...}. This matters most for embed_query and
# fetch_history, which run in parallel — if both returned a full copy of state,
# LangGraph would see two writes to the same unchanged keys (like video_id) in the
# same step and raise InvalidUpdateError. Returning only the changed keys avoids
# that conflict, and LangGraph merges each node's partial update into the state
# automatically (this is the standard pattern for LangGraph nodes, not just a fix
# for parallel branches).
# ─────────────────────────────────────────────

def node_start(state: QAState) -> dict:
    """
    What this does:
        A simple start node that serves as the root of the graph.
        It branches out to run query embedding and history fetching in parallel.
        Returns nothing new — it exists only to give the graph a single entry point
        before it fans out into two parallel branches.
    """
    return {}


def node_embed_query(state: QAState) -> dict:
    """
    What this does:
        Generates a vector embedding of the user's question.
        Uses 'task_type="retrieval_query"' for optimal semantic matching (different from
        "retrieval_document", which is used when embedding video chunks for storage).
    """
    if state.get("error_message"):
        return {}

    try:
        # We pass a list containing the single question and retrieve the first resulting vector
        embeddings = generate_embeddings([state["question"]], task_type="retrieval_query")
        return {"query_embedding": embeddings[0]}
    except Exception as e:
        return {"error_message": f"[node_embed_query] {str(e)}"}


def node_fetch_history(state: QAState) -> dict:
    """
    What this does:
        Fetches the previous chat messages for this session from the database.
    """
    if state.get("error_message"):
        return {}

    try:
        history = get_chat_history(session_id=state["session_id"], video_id=state["video_id"])
        return {"chat_history": history}
    except Exception as e:
        return {"error_message": f"[node_fetch_history] {str(e)}"}


def node_query_chromadb(state: QAState) -> dict:
    """
    What this does:
        Acts as the join/sync point for the parallel nodes.
        Queries ChromaDB using the question embedding to retrieve relevant text chunks from the video.
    """
    if state.get("error_message"):
        return {}

    try:
        context = query_similar_chunks(
            query_embedding=state["query_embedding"],
            video_id=state["video_id"],
            top_k=5
        )
        return {"context": context}
    except Exception as e:
        return {"error_message": f"[node_query_chromadb] {str(e)}"}


def node_llm_answer(state: QAState) -> dict:
    """
    What this does:
        Uses the retrieved context (chunks) and previous history to formulate an answer using Gemini.
    """
    if state.get("error_message"):
        return {}

    try:
        answer = generate_qa_answer(
            question=state["question"],
            context=state["context"],
            history=state["chat_history"] or ""
        )
        return {"answer": answer}
    except Exception as e:
        return {"error_message": f"[node_llm_answer] {str(e)}"}


def node_save_chat(state: QAState) -> dict:
    """
    What this does:
        Saves the current question and the generated answer into the database's chat history.

    Why it exists:
        So that subsequent questions in this session have access to this conversation turn.
        save_chat_message() generates its own embedding internally (see vector_store.py),
        so we only need to pass the plain text here — no vector to compute or pass ourselves.
    """
    if state.get("error_message"):
        return {}

    try:
        # Save user question
        save_chat_message(
            session_id=state["session_id"],
            video_id=state["video_id"],
            role="user",
            message=state["question"]
        )
        # Save assistant answer
        save_chat_message(
            session_id=state["session_id"],
            video_id=state["video_id"],
            role="assistant",
            message=state["answer"]
        )
        return {}
    except Exception as e:
        return {"error_message": f"[node_save_chat] {str(e)}"}


# ─────────────────────────────────────────────
# ROUTING FUNCTION
# ─────────────────────────────────────────────

def check_for_error(state: QAState) -> str:
    """
    What this does:
        Inspects the state and routes to END immediately if an error occurred.
    """
    if state.get("error_message"):
        return "error"
    return "continue"


# ─────────────────────────────────────────────
# GRAPH ASSEMBLY (Parallel Architecture)
# ─────────────────────────────────────────────

def build_qa_graph():
    """
    What this does:
        Assembles nodes into a StateGraph with parallel execution.
        The graph maps:
                    ┌──→ embed_query ───┐
            start ──┤                   ├──→ query_chromadb → llm_answer → save_chat → END
                    └──→ fetch_history ─┘
    """
    graph = StateGraph(QAState)

    # Register nodes
    graph.add_node("start", node_start)
    graph.add_node("embed_query", node_embed_query)
    graph.add_node("fetch_history", node_fetch_history)
    graph.add_node("query_chromadb", node_query_chromadb)
    graph.add_node("llm_answer", node_llm_answer)
    graph.add_node("save_chat", node_save_chat)

    # Entry point
    graph.set_entry_point("start")

    # Branch out parallel processes (start -> embed_query AND start -> fetch_history)
    graph.add_edge("start", "embed_query")
    graph.add_edge("start", "fetch_history")

    # Fan-in (Merge point): Both parallel tasks must finish before query_chromadb executes
    graph.add_conditional_edges(
        "embed_query",
        check_for_error,
        {"continue": "query_chromadb", "error": END}
    )
    graph.add_conditional_edges(
        "fetch_history",
        check_for_error,
        {"continue": "query_chromadb", "error": END}
    )

    # Sequential edges for the remainder of the pipeline
    graph.add_conditional_edges(
        "query_chromadb",
        check_for_error,
        {"continue": "llm_answer", "error": END}
    )
    graph.add_conditional_edges(
        "llm_answer",
        check_for_error,
        {"continue": "save_chat", "error": END}
    )

    graph.add_edge("save_chat", END)

    return graph.compile()


# Compile the QA graph
qa_app = build_qa_graph()