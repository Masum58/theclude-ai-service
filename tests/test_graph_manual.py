from app.graphs.video_processing_graph import video_processing_app
import time
import urllib.parse

filename = "What is Capitulation in Crypto Trading (And How to Respond To Sell-Offs).mp4"
encoded_filename = urllib.parse.quote(filename)
VIDEO_URL = f"http://localhost:8080/{encoded_filename}"

initial_state = {
    "video_id": "test_video_001",
    "video_url": VIDEO_URL,
    "local_video_path": None,
    "local_audio_path": None,
    "transcript": None,
    "chunks": None,
    "embeddings": None,
    "status": "processing",
    "error_message": None,
}

print(f"Starting graph with video_url: {VIDEO_URL}")
print("Running full pipeline (this will take ~1 minute for transcription)...\n")

start_time = time.time()
final_state = video_processing_app.invoke(initial_state)
elapsed = time.time() - start_time

print(f"\n{'='*50}")
print(f"Pipeline finished in {elapsed:.2f} seconds")
print(f"Final status: {final_state['status']}")

if final_state["status"] == "failed":
    print(f"❌ Error: {final_state['error_message']}")
else:
    print(f"✅ Success!")
    print(f"   Video path: {final_state['local_video_path']}")
    print(f"   Audio path: {final_state['local_audio_path']}")
    print(f"   Transcript segments: {len(final_state['transcript'])}")
    print(f"   Chunks: {len(final_state['chunks'])}")
    print(f"   Embeddings: {len(final_state['embeddings'])} vectors, {len(final_state['embeddings'][0])} dimensions each")