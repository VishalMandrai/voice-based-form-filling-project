from scipy.signal import resample
import numpy as np
import streamlit as st
import time

from scipy.io import wavfile
from scipy.signal import resample

def new():
    # 2️⃣ Load generated WAV
    sample_rate, audio = wavfile.read("./src/sample_audio.wav")

    print(sample_rate, audio, audio.shape, audio.dtype, audio.ndim, audio.min(), audio.max())

    # 3️⃣ Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    print(sample_rate, audio, audio.shape, audio.dtype, audio.ndim, audio.min(), audio.max())

    # 4️⃣ Convert to float32 in [-1, 1]
    audio = audio.astype(np.float32)

    print(sample_rate, audio, audio.shape, audio.dtype, audio.ndim, audio.min(), audio.max())

    if audio.dtype == np.int16:
        audio = audio / 32768.0

    # 5️⃣ Resample to 16kHz if needed
    if sample_rate != 16000:
        num_samples = int(len(audio) * 16000 / sample_rate)
        audio = resample(audio, num_samples)
        sample_rate = 16000

    # Final audio array
    audio_np = audio

    print(audio_np)
    print("Final shape:", audio_np.shape)
    print("Sample rate:", sample_rate)
    print("Dtype:", audio_np.dtype)
    print("Min/Max:", audio_np.min(), audio_np.max())




# ─────────────────────────────────────────────────────────────────────────────
# Transcription by Whisper Model
# ─────────────────────────────────────────────────────────────────────────────
def transcribe_audio(frames: list, sample_rate: int = 16000, model = None) -> str:
    """
    Accepts a list of numpy int16 arrays, writes them as a WAV file,
    and transcribes with Whisper.
    Swap this function body to use any other STT service.
    """
    if not frames:
        return "(no audio captured)"

    if model is None:
        return "⚠️ Whisper not installed. Run: pip install openai-whisper"

    # Flatten & normalise to float32 in [-1, 1]
    #sample_rate, frames = wavfile.read("./src/sample_audio.wav")
    
    print("After loading - Sample Rate and Frames Info")
    #print(f"Frames: {frames.shape}, sample rate: {sample_rate}")
    #print("Frame Info: ", type(frames), len(frames), type(frames[0]), frames[0].shape, frames[0].dtype)
    
    audio_np = np.concatenate(frames, axis=0)   ## enable this line when taking real-time input from mic
        
    audio_np = audio_np.astype(np.float32) 
    audio_np = audio_np/ 32768.0
    #audio_np = frames.astype(np.float32) / 32768.0

    num_samples = int(len(audio_np) * 16000 / sample_rate)
    audio_np = resample(audio_np, num_samples)
    
    audio_np = np.ascontiguousarray(audio_np)   # making 'audio_np' array contiguous
    
    print("After processing - Audio_np Info")
    print(f"Audio_np array shape: {audio_np.shape}, sample rate: {sample_rate}")
    print(f"Audio_np array dtype: {audio_np.dtype}, min: {audio_np.min()}, max: {audio_np.max()}")
    
    audio_duration = len(audio_np)/16000
    
    info_1 = str(audio_np.dtype) + ", " + str(audio_np.shape) + ", " + str(audio_np.min()) + ", " + str(audio_np.max()) + "Duration :"  + str(audio_duration) + " seconds"
    
    ## Whisper model expects at least 1 second of audio, so we pad if too short:
    target_len = int(1.0 * 24000)               # 1.5 second; ensuring atleast 1.5 secs of audio length
    if len(audio_np) < target_len:
        pad_amount = target_len - len(audio_np)
        audio_np = np.pad(audio_np, (0, pad_amount))
        
    # 4️⃣ Normalize amplitude (important)
    # audio_np = audio_np / (np.max(np.abs(audio_np)) + 1e-9)
    
    audio_duration = len(audio_np)/16000
    info_2 = str(audio_np.dtype) + ", " + str(audio_np.shape) + ", " + str(audio_np.min()) + ", " + str(audio_np.max()) + "Duration :"  + str(audio_duration) + " seconds"
        
    try:
        result = model.transcribe(audio_np, language = None,  # auto-detect Hindi/English
                                  fp16=False)
        st.write(f"Whisper transcription result: {result}")
        time.sleep(5)  # small delay to ensure result is ready before accessing
        #result_text = result["text"].strip()
        #if not result_text:
        #    return "Audio captured but NO Speech detected!"
        return str(result) + "\n\n" + info_1 + "\n\n" + info_2
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return f"Error during transcription: {e}"
    

    # Write temporary WAV
    #import scipy.io.wavfile as wav
    #with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
    #    tmp_path = f.name
    #    st.write(f"Writing audio to temporary file: {tmp_path}")
    #wav.write(tmp_path, sample_rate, audio_np)

    #try:
    #    result = model.transcribe(tmp_path, fp16=False) 
    #    return result["text"].strip()
    #finally:
    #    os.unlink(tmp_path)  
