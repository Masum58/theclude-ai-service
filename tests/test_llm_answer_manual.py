# WHAT: Manual test for generate_qa_answer() and generate_video_summary() using real video content
# WHY: Verify Gemini LLM calls work correctly with the real chunked transcript as context

from app.nodes.audio_extractor import extract_audio
from app.nodes.transcriber import transcribe_audio
from app.nodes.text_chunker import chunk_transcript
from app.nodes.llm_answer import generate_qa_answer, generate_video_summary
import time

LOCAL_VIDEO_PATH = r"C:\Users\masum\Downloads\Televzr Downloads\What is Capitulation in Crypto Trading (And How to Respond To Sell-Offs).mp4"

print("Preparing context (extract -> transcribe -> chunk)...")
audio_path = extract_audio(LOCAL_VIDEO_PATH)
segments = transcribe_audio(audio_path)
chunks = chunk_transcript(segments, chunk_size=1000)
print(f"✅ Got {len(chunks)} chunks ready as context")

# চলো একটা প্রাসঙ্গিক প্রশ্ন করি, যেটার উত্তর transcript-এ আছে
question = "What is capitulation in crypto trading?"
context = "\n\n".join(chunks)  # সব chunk একসাথে জুড়ে দিলাম (আসল pipeline এ শুধু relevant chunk যাবে ChromaDB থেকে)

print(f"\nAsking: '{question}'")
start_time = time.time()
try:
    answer = generate_qa_answer(question=question, context=context, history="")
    elapsed = time.time() - start_time
    print(f"✅ Success! Answer generated in {elapsed:.2f}s\n")
    print(f"Answer:\n{answer}")
except Exception as e:
    print(f"❌ Failed: {e}")

print("\n\nTesting with empty question...")
try:
    generate_qa_answer(question="", context=context)
except ValueError as e:
    print(f"✅ Correctly raised ValueError: {e}")

print("\nTesting summary generation...")
try:
    full_transcript = " ".join([s["text"] for s in segments])
    summary = generate_video_summary(full_transcript)
    print(f"✅ Summary generated:\n{summary}")
except Exception as e:
    print(f"❌ Failed: {e}")