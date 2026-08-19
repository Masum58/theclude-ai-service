# WHAT: Manual test for the /ask-video-stream endpoint
# WHY: Verify chunks arrive progressively (not all at once) and chat history gets saved after
import requests
import time

BASE_URL = "http://127.0.0.1:8001/api/v1/video"
video_id = "api_test_001"  # আগে থেকে processed একটা video_id ব্যবহার করছি

print("Asking a streaming question...\n")
start = time.time()
first_chunk_time = None

with requests.post(
    f"{BASE_URL}/ask-video-stream",
    json={"video_id": video_id, "question": "What is capitulation in crypto trading?"},
    stream=True
) as response:
    print(f"Status code: {response.status_code}")
    print(f"Session ID header: {response.headers.get('X-Session-Id')}\n")

    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
        if first_chunk_time is None:
            first_chunk_time = time.time()
            print(f"[First chunk arrived after {first_chunk_time - start:.2f}s]\n")
        print(chunk, end="", flush=True)

total_time = time.time() - start
print(f"\n\n[Total time: {total_time:.2f}s]")