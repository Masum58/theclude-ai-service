# WHAT: Manual test script for transcribe_audio() node
# WHY: Verify Whisper transcription works correctly before wiring into LangGraph
# আগের সফল টেস্ট থেকে পাওয়া .wav ফাইল সরাসরি ব্যবহার করছি
# (sample-videos.com অনির্ভরযোগ্য, তাই download ধাপ এড়িয়ে যাচ্ছি)

from app.nodes.transcriber import transcribe_audio
import time
import os

# আগের audio_extractor টেস্ট থেকে পাওয়া ফাইল — তোমার actual filename দিয়ে বদলে নাও
EXISTING_AUDIO_PATH = r"C:\Users\masum\AppData\Local\Temp\audio_364b4203620741d8b14f93819f819267.wav"

print("Checking if the audio file still exists...")
if not os.path.exists(EXISTING_AUDIO_PATH):
    print(f"❌ File not found: {EXISTING_AUDIO_PATH}")
    print("   Run test_audio_extraction_manual.py first to generate a fresh .wav file.")
else:
    print(f"✅ Found audio file: {EXISTING_AUDIO_PATH}")

    print("\nTranscribing audio (this may take a moment, first run downloads the model)...")
    start_time = time.time()
    try:
        segments = transcribe_audio(EXISTING_AUDIO_PATH)
        elapsed = time.time() - start_time
        print(f"✅ Success! Transcription took {elapsed:.2f} seconds")
        print(f"   Total segments: {len(segments)}")
        print("\n   First 5 segments:")
        for seg in segments[:5]:
            print(f"   [{seg['start']}s - {seg['end']}s] {seg['text']}")
    except Exception as e:
        print(f"❌ Failed: {e}")

print("\nTesting with an empty path...")
try:
    transcribe_audio("")
except ValueError as e:
    print(f"✅ Correctly raised ValueError: {e}")

print("\nTesting with a non-existent file...")
try:
    transcribe_audio("C:\\fake\\path\\doesnotexist.wav")
except FileNotFoundError as e:
    print(f"✅ Correctly raised FileNotFoundError: {e}")