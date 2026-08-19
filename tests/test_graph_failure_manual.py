# WHAT: Test the failure path of video_processing_graph — an invalid URL should
#       fail at download_video and skip all downstream nodes cleanly
from app.graphs.video_processing_graph import video_processing_app

initial_state = {
    "video_id": "test_fail_001",
    "video_url": "https://this-domain-does-not-exist-12345.com/fake.mp4",
    "local_video_path": None,
    "local_audio_path": None,
    "transcript": None,
    "chunks": None,
    "embeddings": None,
    "status": "processing",
    "error_message": None,
}

final_state = video_processing_app.invoke(initial_state)

print(f"Final status: {final_state['status']}")
print(f"Error message: {final_state['error_message']}")
print(f"local_audio_path (should still be None): {final_state['local_audio_path']}")
print(f"chunks (should still be None): {final_state['chunks']}")