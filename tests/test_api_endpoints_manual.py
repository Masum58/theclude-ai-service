# WHAT: Manual test for /upload, /status, and /ask-video HTTP endpoints (requires the server to be running)
# WHY: Verify the full API flow — submit a job, poll until it finishes, then ask a question about it
# চালানোর আগে: uvicorn app.main:app --reload --reload-dir app --port 8001 আলাদা টার্মিনালে চালু রাখো
# এবং video folder-এ python -m http.server 8080 আরেকটা টার্মিনালে

import requests
import time
import urllib.parse

BASE_URL = "http://127.0.0.1:8001/api/v1/video"
filename = "What is Capitulation in Crypto Trading (And How to Respond To Sell-Offs).mp4"
video_url = f"http://localhost:8080/{urllib.parse.quote(filename)}"
video_id = "api_test_001"

print("Step 1: Submitting video for processing...")
response = requests.post(f"{BASE_URL}/upload", json={
    "video_id": video_id,
    "video_url": video_url
})
print(f"Status code: {response.status_code}")
print(f"Response: {response.json()}\n")

print("Step 2: Polling status every 5 seconds until completed or failed...")
max_attempts = 20
for attempt in range(max_attempts):
    time.sleep(5)
    status_response = requests.get(f"{BASE_URL}/status/{video_id}")
    status_data = status_response.json()
    print(f"[Attempt {attempt + 1}] Status: {status_data['status']}")

    if status_data["status"] == "completed":
        print(f"\n✅ Success! Chunks stored: {status_data['chunks_stored']}")
        break
    elif status_data["status"] == "failed":
        print(f"\n❌ Failed: {status_data['error_message']}")
        break
else:
    print("\n⚠️ Timed out waiting for completion.")

print("\nStep 3: Testing status check for a non-existent video_id...")
not_found_response = requests.get(f"{BASE_URL}/status/does_not_exist_999")
print(f"Status code (should be 404): {not_found_response.status_code}")
print(f"Response: {not_found_response.json()}")

print("\n\nStep 4: Asking a question about the processed video...")
qa_response = requests.post(f"{BASE_URL}/ask-video", json={
    "video_id": video_id,
    "question": "What is capitulation in crypto trading?"
})
qa_data = qa_response.json()
print(f"Status code: {qa_response.status_code}")
print(f"Session ID: {qa_data.get('session_id')}")
print(f"Answer: {qa_data.get('answer')}")
print(f"Error message: {qa_data.get('error_message')}")

print("\nStep 5: Asking a follow-up question in the SAME session (tests chat history)...")
followup_response = requests.post(f"{BASE_URL}/ask-video", json={
    "video_id": video_id,
    "session_id": qa_data.get("session_id"),
    "question": "Is that a good time to sell?"
})
followup_data = followup_response.json()
print(f"Status code: {followup_response.status_code}")
print(f"Session ID (should match Step 4): {followup_data.get('session_id')}")
print(f"Answer: {followup_data.get('answer')}")

print("\nStep 6: Asking about a video that was never uploaded (should 404)...")
bad_video_response = requests.post(f"{BASE_URL}/ask-video", json={
    "video_id": "nonexistent_video_999",
    "question": "Anything?"
})
print(f"Status code (should be 404): {bad_video_response.status_code}")
print(f"Response: {bad_video_response.json()}")

print("\nStep 7: Asking about a video that is still processing (should 409)...")
# একটা নতুন ভিডিও submit করে সাথে সাথেই প্রশ্ন করছি, processing শেষ হওয়ার আগেই
still_processing_id = "api_test_002_still_processing"
requests.post(f"{BASE_URL}/upload", json={
    "video_id": still_processing_id,
    "video_url": video_url
})
immediate_qa_response = requests.post(f"{BASE_URL}/ask-video", json={
    "video_id": still_processing_id,
    "question": "Anything?"
})
print(f"Status code (should be 409): {immediate_qa_response.status_code}")
print(f"Response: {immediate_qa_response.json()}")