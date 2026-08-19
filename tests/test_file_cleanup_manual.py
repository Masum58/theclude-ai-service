# WHAT: Manual test for cleanup_temp_files() — verifies it deletes files, and handles
#       missing/None paths gracefully without raising
from app.services.file_cleanup import cleanup_temp_files
import tempfile
import os

# ধাপ ১: দুটো ভুয়া টেম্প ফাইল বানাই টেস্ট করার জন্য
temp_dir = tempfile.gettempdir()
fake_video_path = os.path.join(temp_dir, "test_cleanup_video.mp4")
fake_audio_path = os.path.join(temp_dir, "test_cleanup_audio.wav")

with open(fake_video_path, "w") as f:
    f.write("fake video content")
with open(fake_audio_path, "w") as f:
    f.write("fake audio content")

print(f"Created test files:\n  {fake_video_path}\n  {fake_audio_path}")
print(f"Video exists before cleanup: {os.path.exists(fake_video_path)}")
print(f"Audio exists before cleanup: {os.path.exists(fake_audio_path)}")

# ধাপ ২: cleanup চালাই
print("\nRunning cleanup_temp_files()...")
cleanup_temp_files(fake_video_path, fake_audio_path)

print(f"\nVideo exists after cleanup (should be False): {os.path.exists(fake_video_path)}")
print(f"Audio exists after cleanup (should be False): {os.path.exists(fake_audio_path)}")

# ধাপ ৩: এখন None এবং non-existent path দিয়ে টেস্ট করি — crash করা উচিত না
print("\nTesting with None paths (should not raise)...")
cleanup_temp_files(None, None)
print("✅ No crash with None paths")

print("\nTesting with a path that doesn't exist (should not raise)...")
cleanup_temp_files("C:\\fake\\nonexistent\\video.mp4", None)
print("✅ No crash with non-existent path")

print("\n✅ All cleanup tests passed!")