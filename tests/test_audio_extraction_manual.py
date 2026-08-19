# WHAT: Manual test script for extract_audio() node
# WHY: Verify FFmpeg audio extraction works before wiring into LangGraph
# প্রথমে download_video() দিয়ে একটা ভিডিও নামাই, তারপর সেটা থেকে audio বের করি

from app.nodes.video_downloader import download_video
from app.nodes.audio_extractor import extract_audio
import os

# audio track আছে এমন sample video (আগেরবার confirm করা হয়েছিল)
SAMPLE_URL = "https://archive.org/download/BigBuckBunny_124/Content/big_buck_bunny_720p_surround.mp4"

print("Step 1: Downloading video...")
video_path = download_video(SAMPLE_URL)
print(f"✅ Video downloaded: {video_path}")

print("\nStep 2: Extracting audio from video...")
try:
    audio_path = extract_audio(video_path)
    print(f"✅ Success! Audio saved at: {audio_path}")
    print(f"   File size: {os.path.getsize(audio_path)} bytes")
except Exception as e:
    print(f"❌ Failed: {e}")

print("\nTesting with an empty path...")
try:
    extract_audio("")
except ValueError as e:
    print(f"✅ Correctly raised ValueError: {e}")

print("\nTesting with a non-existent file...")
try:
    extract_audio("C:\\fake\\path\\doesnotexist.mp4")
except FileNotFoundError as e:
    print(f"✅ Correctly raised FileNotFoundError: {e}")