# WHAT: Manual test for qa_graph.py — full Q&A pipeline with parallel embed+history fetch
# WHY: Verify parallel nodes, ChromaDB retrieval, LLM answer, and chat history saving all work together
# ব্যবহার করছি সেই video_id যেটাতে আগে data store করা হয়েছিল (test_graph_manual.py থেকে)

from app.graphs.qa_graph import qa_app

initial_state = {
    "video_id": "test_video_001",
    "session_id": "test_session_001",
    "question": "What is capitulation in crypto trading?",
    "query_embedding": None,
    "chat_history": None,
    "context": None,
    "answer": None,
    "error_message": None,
}

print(f"Asking: '{initial_state['question']}'")
final_state = qa_app.invoke(initial_state)

if final_state.get("error_message"):
    print(f"❌ Error: {final_state['error_message']}")
else:
    print(f"✅ Success!\n")
    print(f"Answer:\n{final_state['answer']}\n")
    print(f"Context length: {len(final_state['context'])} chars")
    print(f"Chat history at time of asking: '{final_state['chat_history']}'")

# ─── দ্বিতীয় প্রশ্ন, একই session — history কাজ করছে কিনা দেখার জন্য ───
print("\n\n--- Asking a follow-up question in the same session ---")
followup_state = {
    "video_id": "test_video_001",
    "session_id": "test_session_001",
    "question": "Is that a good time to sell?",
    "query_embedding": None,
    "chat_history": None,
    "context": None,
    "answer": None,
    "error_message": None,
}
final_followup = qa_app.invoke(followup_state)

if final_followup.get("error_message"):
    print(f"❌ Error: {final_followup['error_message']}")
else:
    print(f"✅ Success!\n")
    print(f"Answer:\n{final_followup['answer']}\n")
    print(f"Chat history used for this answer:\n{final_followup['chat_history']}")