import whisper
import pyaudio              ## Use 'sounddevice' instead of pyaudio for better performance and stability
import numpy as np
import queue
import threading
import time
import torch

# -------------------------------
# Configuration
# -------------------------------
MODEL_SIZE = "small"
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 2  # Process every 2 seconds

# -------------------------------
# Load Whisper Model
# -------------------------------
#device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on...")
model = whisper.load_model(MODEL_SIZE)#.to(device)

# -------------------------------
# Audio Queue
# -------------------------------
audio_queue = queue.Queue()

# -------------------------------
# Audio Capture Callback
# -------------------------------
def callback(in_data, frame_count, time_info, status):
    audio_queue.put(in_data)
    return (in_data, pyaudio.paContinue)

# -------------------------------
# Start Microphone Stream
# -------------------------------
p = pyaudio.PyAudio()

stream = p.open(format=pyaudio.paInt16,
                channels=2,
                rate=RATE,
                input=True, input_device_index=1,
                frames_per_buffer=CHUNK,
                stream_callback=callback)

stream.start_stream()

print("🎙️ Listening... (Ctrl+C to stop)")

# -------------------------------
# Real-Time Processing Loop
# -------------------------------
try:
    while True:
        frames = []
        start_time = time.time()

        # Collect audio for fixed duration
        while time.time() - start_time < RECORD_SECONDS:
            print("inside detecting and printing loop")
            if not audio_queue.empty():
                data = audio_queue.get()
                frames.append(np.frombuffer(data, np.int16))

        if frames:
            print("Audio captured, processing...")
            audio_np = np.concatenate(frames).astype(np.float32) / 32768.0
            print("Processing audio...", audio_np)

            result = model.transcribe(
                audio_np,
                language=None,  # auto-detect Hindi/English
                fp16=torch.cuda.is_available()
            )

            print(result)
            print("📝", result["text"])

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()