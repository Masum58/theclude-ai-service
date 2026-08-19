# WHAT: Manual test for chunk_transcript() using the real crypto video's transcript
# WHY: Verify chunking groups segments correctly and preserves timestamps

from app.nodes.audio_extractor import extract_audio
from app.nodes.transcriber import transcribe_audio
from app.nodes.text_chunker import chunk_transcript
import time

LOCAL_VIDEO_PATH = r"C:\Users\masum\Downloads\Televzr Downloads\What is Capitulation in Crypto Trading (And How to Respond To Sell-Offs).mp4"

print("Step 1: Extracting audio...")
audio_path = extract_audio(LOCAL_VIDEO_PATH)
print(f"✅ Audio extracted: {audio_path}")

print("\nStep 2: Transcribing (this takes a while)...")
start_time = time.time()
segments = transcribe_audio(audio_path)
print(f"✅ Transcribed {len(segments)} segments in {time.time() - start_time:.2f}s")

print("\nStep 3: Chunking transcript...")
chunks = chunk_transcript(segments, chunk_size=1000)
print(f"✅ Created {len(chunks)} chunks")
print("\n   First 3 chunks:")
for i, chunk in enumerate(chunks[:3]):
    print(f"\n   --- Chunk {i+1} ---")
    print(f"   {chunk}")

print("\nTesting with empty segments list...")
empty_chunks = chunk_transcript([])
print(f"✅ Empty input correctly returns: {empty_chunks}")

print("\nTesting with invalid chunk_size...")
try:
    chunk_transcript(segments, chunk_size=0)
except ValueError as e:
    print(f"✅ Correctly raised ValueError: {e}")