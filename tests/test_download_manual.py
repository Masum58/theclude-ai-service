# WHAT: Manual test script for download_video() node
# WHY: Verify the node works correctly before wiring into LangGraph
# একটা পাবলিক sample video URL দিয়ে টেস্ট করছি

from app.nodes.video_downloader import download_video

# একটা ছোট, publicly available sample video (Google-এর টেস্ট ভিডিও)
SAMPLE_URL = "https://www.w3schools.com/html/mov_bbb.mp4"
print("Testing download_video() with a valid URL...")
try:
    path = download_video(SAMPLE_URL)
    print(f"✅ Success! Video saved at: {path}")
    import os
    print(f"   File size: {os.path.getsize(path)} bytes")
except Exception as e:
    print(f"❌ Failed: {e}")

print("\nTesting with an invalid (empty) URL...")
try:
    download_video("")
except ValueError as e:
    print(f"✅ Correctly raised ValueError: {e}")

print("\nTesting with a malformed URL (no http)...")
try:
    download_video("not-a-url")
except ValueError as e:
    print(f"✅ Correctly raised ValueError: {e}")

print("\nTesting with an unreachable URL...")
try:
    download_video("https://this-domain-does-not-exist-12345.com/video.mp4")
except ConnectionError as e:
    print(f"✅ Correctly raised ConnectionError: {e}")