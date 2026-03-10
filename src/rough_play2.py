import pyaudio
import torch
import whisper

p = pyaudio.PyAudio()

for i in range(p.get_device_count()):
    device = p.get_device_info_by_index(i)
    print(i, device["name"], "Input Channels:", device["maxInputChannels"])

whisper_model = whisper.load_model("small")
result = whisper_model.transcribe("./src/audio_test.m4a", 
                                  language=None,  # auto-detect Hindi/English
                                  fp16=torch.cuda.is_available())
print(result["text"])
    
with open("./src/audio_test.m4a", "rb") as file:
    whisper_model = whisper.load_model("small")
    result = whisper_model.transcribe(file)
    print(result["text"])