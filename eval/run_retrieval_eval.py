# WHAT: Measures retrieval quality — for each golden question, checks whether the expected
#       chunk (by timestamp range) appears in the top-5 chunks ChromaDB returns.
# WHY: Separates "did we find the right context?" from "did the LLM write a good answer?" —
#      a bad answer caused by bad retrieval needs a different fix than a bad answer caused
#      by bad generation.
# INPUT: eval/golden_dataset.json
# OUTPUT: eval/results/retrieval_eval_<timestamp>.json + printed Hit Rate

import json
from datetime import datetime

from app.nodes.embedder import generate_embeddings
from app.nodes.vector_store import video_chunks_collection

GOLDEN_DATASET_PATH = "eval/golden_dataset.json"
TOP_K = 5


def load_golden_dataset():
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["entries"]


def run_retrieval_eval():
    entries = load_golden_dataset()
    results = []
    hits = 0

    print(f"Running retrieval evaluation on {len(entries)} questions...\n")

    for entry in entries:
        question = entry["question"]
        video_id = entry["video_id"]
        expected_range = entry["expected_chunk_range"]

        query_embedding = generate_embeddings([question], task_type="retrieval_query")[0]

        raw_results = video_chunks_collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K,
            where={"video_id": video_id},
        )
        retrieved_docs = raw_results.get("documents", [[]])[0]

        is_hit = any(doc.startswith(expected_range) for doc in retrieved_docs)
        if is_hit:
            hits += 1

        results.append({
            "id": entry["id"],
            "question": question,
            "expected_chunk_range": expected_range,
            "hit": is_hit,
            "retrieved_ranges": [doc[:doc.find("]") + 1] for doc in retrieved_docs],
        })

        status = "✅ HIT" if is_hit else "❌ MISS"
        print(f"[{entry['id']}] {status}  — {question}")
        if not is_hit:
            print(f"    expected: {expected_range}")
            print(f"    got:      {results[-1]['retrieved_ranges']}")

    hit_rate = hits / len(entries) if entries else 0

    print(f"\n{'='*50}")
    print(f"Hit Rate: {hits}/{len(entries)} = {hit_rate:.1%}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"eval/results/retrieval_eval_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "hit_rate": hit_rate,
            "hits": hits,
            "total": len(entries),
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved detailed results to: {output_path}")


if __name__ == "__main__":
    run_retrieval_eval()