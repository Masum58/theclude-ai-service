# WHAT: Manual test using a real local video file (not downloaded from URL)
# WHY: Test extract_audio() + transcribe_audio() with real human speech content
# এখানে video_downloader.py বাদ দিচ্ছি কারণ ফাইলটা already local disk-এ আছে

from app.nodes.audio_extractor import extract_audio
from app.nodes.transcriber import transcribe_audio
import time

# তোমার আসল crypto education ভিডিও
LOCAL_VIDEO_PATH = r"C:\Users\masum\Downloads\Televzr Downloads\What is Capitulation in Crypto Trading (And How to Respond To Sell-Offs).mp4"

print("Step 1: Extracting audio from local video...")
audio_path = extract_audio(LOCAL_VIDEO_PATH)
print(f"✅ Audio extracted: {audio_path}")

print("\nStep 2: Transcribing audio (this may take a moment)...")
start_time = time.time()
segments = transcribe_audio(audio_path)
elapsed = time.time() - start_time

print(f"✅ Success! Transcription took {elapsed:.2f} seconds")
print(f"   Total segments: {len(segments)}")
print("\n   Full transcript:")
for seg in segments:
    print(f"   [{seg['start']}s - {seg['end']}s] {seg['text']}")