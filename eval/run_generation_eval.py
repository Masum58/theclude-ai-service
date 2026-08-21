# WHAT: Evaluates generation quality — checks if LLM answers contain expected key facts (Faithfulness & Fact Recall)
# WHY: Verifies that the model generates accurate, grounded answers with automatic 429 rate-limit handling
# INPUT: eval/golden_dataset.json + ChromaDB knowledge base
# OUTPUT: eval/results/generation_eval_<timestamp>.json + printed Fact Recall scores

import json
import time
from datetime import datetime
import google.generativeai as genai
from app.graphs.qa_graph import qa_app
from app.core.config import settings

GOLDEN_DATASET_PATH = "eval/golden_dataset.json"

# Configure Gemini for evaluation judge
genai.configure(api_key=settings.GEMINI_API_KEY)
judge_model = genai.GenerativeModel("gemini-3.5-flash")


def load_golden_dataset():
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["entries"]


def invoke_qa_with_retry(video_id: str, session_id: str, question: str, max_retries: int = 5) -> str:
    """
    Invokes the QA graph with automatic backoff retry on 429 rate-limit errors.
    """
    for attempt in range(max_retries):
        state = qa_app.invoke({
            "video_id": video_id,
            "session_id": session_id,
            "question": question,
            "query_embedding": None,
            "chat_history": None,
            "context": None,
            "answer": None,
            "error_message": None
        })
        
        error_msg = state.get("error_message")
        if error_msg and ("429" in error_msg or "quota" in error_msg.lower()):
            wait_time = 12 + attempt * 5
            print(f"    ⏳ Rate limit (429) hit on answer generation. Waiting {wait_time}s before retry (attempt {attempt+1}/{max_retries})...")
            time.sleep(wait_time)
            continue
            
        if state.get("answer"):
            return state["answer"]
            
        if error_msg:
            print(f"    ⚠️ Error: {error_msg}")
            return "No answer generated"

    return "No answer generated (rate limit exceeded after retries)"


def judge_fact_coverage(question: str, answer: str, expected_key_facts: list, max_retries: int = 5) -> dict:
    """
    Uses Gemini as a judge with retry on 429 rate-limit errors.
    """
    if not answer or answer.startswith("No answer generated"):
        return {
            "covered_facts": [],
            "missing_facts": expected_key_facts,
            "score": 0.0
        }

    facts_str = "\n".join([f"- {fact}" for fact in expected_key_facts])
    
    prompt = f"""You are an automated evaluation judge for a question-answering system.

Question: {question}

Generated Answer:
{answer}

Expected Key Facts:
{facts_str}

Evaluate which of the expected key facts are present or accurately conveyed in the generated answer.
Return a JSON object with this EXACT structure:
{{
  "covered_facts": ["fact 1 that was mentioned", ...],
  "missing_facts": ["fact that was missing", ...],
  "score": 0.85
}}
Do NOT include markdown fences, return valid JSON only."""

    for attempt in range(max_retries):
        try:
            response = judge_model.generate_content(prompt)
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            return json.loads(text.strip())
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                wait_time = 12 + attempt * 5
                print(f"    ⏳ Rate limit (429) hit on judge. Waiting {wait_time}s before retry (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait_time)
                continue
            
            # Fallback simple keyword match
            matched = [fact for fact in expected_key_facts if any(word.lower() in answer.lower() for word in fact.split()[:2])]
            missing = [fact for fact in expected_key_facts if fact not in matched]
            score = len(matched) / len(expected_key_facts) if expected_key_facts else 0
            return {
                "covered_facts": matched,
                "missing_facts": missing,
                "score": round(score, 2),
                "fallback_used": True,
                "judge_error": str(e)
            }

    return {
        "covered_facts": [],
        "missing_facts": expected_key_facts,
        "score": 0.0
    }


def run_generation_eval():
    entries = load_golden_dataset()
    results = []
    total_score = 0.0

    print(f"Running generation evaluation on {len(entries)} questions with auto 429 retry...\n")

    for entry in entries:
        q_id = entry["id"]
        question = entry["question"]
        video_id = entry["video_id"]
        expected_facts = entry["expected_key_facts"]

        # Run QA Graph with retry
        answer = invoke_qa_with_retry(
            video_id=video_id,
            session_id=f"eval_session_{q_id}",
            question=question
        )

        eval_result = judge_fact_coverage(question, answer, expected_facts)
        score = eval_result.get("score", 0.0)
        total_score += score

        results.append({
            "id": q_id,
            "question": question,
            "video_id": video_id,
            "answer": answer,
            "expected_key_facts": expected_facts,
            "covered_facts": eval_result.get("covered_facts", []),
            "missing_facts": eval_result.get("missing_facts", []),
            "score": score
        })

        print(f"[{q_id}] Score: {score*100:.1f}% — {question}")
        print(f"    Answer snippet: {answer[:120]}...\n")

    avg_score = total_score / len(entries) if entries else 0.0

    print(f"{'='*50}")
    print(f"Average Fact Recall / Quality Score: {avg_score*100:.1f}%")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"eval/results/generation_eval_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "average_score": avg_score,
            "total_questions": len(entries),
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved detailed generation results to: {output_path}")


if __name__ == "__main__":
    run_generation_eval()
