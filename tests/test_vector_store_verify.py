# WHAT: Verifies that store_video_chunks() actually persisted data by querying it back
# WHY: The graph reported "completed", but we should confirm data survives in ChromaDB

from app.nodes.vector_store import query_similar_chunks
from app.nodes.embedder import generate_embeddings

video_id = "test_video_001"
question = "What is capitulation?"

print(f"Querying ChromaDB for video_id='{video_id}'...")
query_vector = generate_embeddings([question], task_type="retrieval_query")[0]

result = query_similar_chunks(query_embedding=query_vector, video_id=video_id, top_k=3)

if result:
    print("✅ Data found in ChromaDB! Retrieved context:\n")
    print(result[:500] + "...")  # প্রথম ৫০০ ক্যারেক্টার দেখাচ্ছি
else:
    print("❌ No data found — either storage didn't happen, or video_id doesn't match.")