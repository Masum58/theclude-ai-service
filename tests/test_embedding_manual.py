# WHAT: Manual test for generate_embeddings() using the real crypto video's chunks
# WHY: Verify Gemini embedding API call works and returns correctly-sized vectors

from app.nodes.audio_extractor import extract_audio
from app.nodes.transcriber import transcribe_audio
from app.nodes.text_chunker import chunk_transcript
from app.nodes.embedder import generate_embeddings
import time

LOCAL_VIDEO_PATH = r"C:\Users\masum\Downloads\Televzr Downloads\What is Capitulation in Crypto Trading (And How to Respond To Sell-Offs).mp4"

print("Step 1: Extracting audio...")
audio_path = extract_audio(LOCAL_VIDEO_PATH)
print(f"✅ Audio extracted")

print("\nStep 2: Transcribing...")
segments = transcribe_audio(audio_path)
print(f"✅ Transcribed {len(segments)} segments")

print("\nStep 3: Chunking...")
chunks = chunk_transcript(segments, chunk_size=1000)
print(f"✅ Created {len(chunks)} chunks")

print("\nStep 4: Generating embeddings (calling Gemini API)...")
start_time = time.time()
try:
    embeddings = generate_embeddings(chunks)
    elapsed = time.time() - start_time
    print(f"✅ Success! Generated {len(embeddings)} embeddings in {elapsed:.2f}s")
    print(f"   Vector dimension (should be 768): {len(embeddings[0])}")
    print(f"   First 5 values of first vector: {embeddings[0][:5]}")
except Exception as e:
    print(f"❌ Failed: {e}")

print("\nTesting with empty chunks list...")
empty_result = generate_embeddings([])
print(f"✅ Empty input correctly returns: {empty_result}")