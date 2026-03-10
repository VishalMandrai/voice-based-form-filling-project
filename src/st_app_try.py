"""
Hold-to-Speak Streamlit App using streamlit-webrtc
====================================================
Install dependencies:
    pip install streamlit streamlit-webrtc av openai-whisper numpy scipy

Run:
    streamlit run app.py

Works on both Desktop and Mobile browsers.
"""

import queue
import threading
import time
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import logging
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

from model_inf import transcribe_audio

# ── Optional: choose your transcription backend ──────────────────────────────
# We use OpenAI Whisper (local, free). To use OpenAI API instead, swap
# transcribe_audio() below for an openai.Audio.transcribe() call.
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎙️ Hold to Speak",
    page_icon="🎙️",
    layout="centered",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — big round button, mobile-friendly
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Hide default Streamlit menu & footer for cleaner look */
    #MainMenu, footer { visibility: hidden; }

    /* Center everything */
    .block-container { max-width: 680px; padding-top: 2rem; }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    .status-idle     { background:#e8f5e9; color:#2e7d32; }
    .status-recording{ background:#ffebee; color:#c62828; }
    .status-processing{ background:#fff3e0; color:#e65100; }

    /* Transcript box */
    .transcript-box {
        background: #f8f9fa;
        border-left: 4px solid #4CAF50;
        border-radius: 8px;
        padding: 16px 20px;
        font-size: 1.05rem;
        line-height: 1.6;
        margin-top: 1rem;
        white-space: pre-wrap;
    }

    /* WebRTC button overrides for mobile tap */
    button { touch-action: manipulation; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────
if "recording" not in st.session_state:
    st.session_state.recording = False          # True - when user records the audio input
if "processing" not in st.session_state:
    st.session_state.processing = False         # True - when we are processing
if "audio_buffer" not in st.session_state:
    st.session_state.audio_buffer = []          # list of numpy arrays
if "sample_rate" not in st.session_state:
    st.session_state.sample_rate = 0            # sample rate of the recorded audio
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "whisper_model" not in st.session_state:
    st.session_state.whisper_model = None

# Creating a separate class of "recording flag" to manage state across threads (especially for callback thread access) without relying on Streamlit's session state, which is not thread-safe for writes.
class RecordingState:
    def __init__(self):
        self.recording = st.session_state.recording
        self.lock = threading.Lock()   # Lock to safely flip the recording flag from both threads

# NOTE: WebRTC callback (i.e., audio_frame_callback) runs in a persistent background thread that does not automatically sync with Streamlit's rerun-based session state updates in the main script.

recording_state = RecordingState()  # Create an instance of the recording state class

################
st.write("\n\n")
st.write("recording flag :", recording_state.recording, st.session_state.recording)
st.write("processing flag :", st.session_state.processing)
st.write("Whisper Model flag :", st.session_state.whisper_model)


# Thread-safe queue: audio frames flow from webrtc callback → main thread
audio_queue: queue.Queue = queue.Queue()
# Lock to safely flip the recording flag from both threads
record_lock = threading.Lock()

# ─────────────────────────────────────────────────────────────────────────────
# Load Whisper model (cached so it only loads once)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Whisper model…")
def load_whisper(model_name: str = "base"):
    if not WHISPER_AVAILABLE:
        print("Whisper is not available.")
        st.write("Whisper is NOT available!")
        return None

    st.write(f"Whisper is Available: Model - {model_name}")
    return whisper.load_model(model_name)


# ─────────────────────────────────────────────────────────────────────────────
# WebRTC audio callback (runs in a separate thread!)
# ─────────────────────────────────────────────────────────────────────────────
def audio_frame_callback(frame: av.AudioFrame) -> av.AudioFrame:
    """
    Called for every audio frame coming from the microphone.
    We only enqueue frames while st.session_state.recording is True.
    Note: st.session_state access from threads is safe for simple reads.
    """
    #print("Entered callback function!")
    with recording_state.lock:
        #print("Entered record lock in callback!")
        is_recording = recording_state.recording
        #print("Existing record lock in Callback!", is_recording)

    if is_recording:
        print("Recording frame, putting into queue")
        # Convert to mono int16 numpy array
        pcm = frame.to_ndarray()          # shape: (channels, samples)
        print("PCM: ", pcm.shape, pcm.ndim, len(pcm))
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=0)        # mix to mono
        pcm = pcm.astype(np.int16)
        print("PCM 2: ", pcm.shape, pcm.ndim, len(pcm))
        
        audio_queue.put((pcm, frame.sample_rate))

    #print("frames are :", frame.to_ndarray() , "with sample rate:", frame.sample_rate)
    #print("frame shape:", frame.to_ndarray().shape)
    return frame                          # must return a frame


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("🎙️ Hold to Speak")
st.caption("Press **Start** below to activate the mic, then use the toggle to record.")

# ── WebRTC streamer (audio only, no video) ────────────────────────────────────
ctx = webrtc_streamer(
    key="hold-to-speak",
    mode=WebRtcMode.SENDONLY,           # we only send audio (no playback)
    audio_frame_callback=audio_frame_callback,
    media_stream_constraints={"audio": {"echoCancellation": True,
                                        "noiseSuppression": True},
                              "video": False},
    rtc_configuration={
        # Public STUN server — required for cloud / mobile deployments
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    async_processing=True,
)

st.divider()

# ── Only show controls once the WebRTC stream is active ──────────────────────
if ctx.state.playing:

    # ── Status badge ─────────────────────────────────────────────────────────
    if st.session_state.processing:
        st.markdown('<span class="status-badge status-processing">⏳ Processing…</span>',
                    unsafe_allow_html=True)
    elif st.session_state.recording:
        st.markdown('<span class="status-badge status-recording">🔴 Recording…</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-idle">✅ Ready</span>',
                    unsafe_allow_html=True)

    # ── Record / Stop buttons ─────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        # Disable while already recording or processing
        start_disabled = recording_state.recording or st.session_state.processing
        #start_disabled = st.session_state.recording
        
        if st.button("🎤 Start Recording", disabled=start_disabled,
                     use_container_width=True, type="primary"):
            with record_lock:
                st.session_state.recording = True
                st.session_state.audio_buffer = []
                # Drain any stale frames from the queue
                while not audio_queue.empty():
                    audio_queue.get_nowait()
            st.rerun()

    with col2:
        stop_disabled = not recording_state.recording
        if st.button("⏹️ Stop & Transcribe", disabled=stop_disabled,
                     use_container_width=True):
            # 1. Stop recording
            with record_lock:
                st.session_state.recording = False

            # 2. Drain the queue into the buffer
            print("Draining Queue in main thread")
            frames, sample_rate = [], 0
            time.sleep(0.1)  # small delay to let the last frames arrive
            while not audio_queue.empty():
                print("Getting frame from queue")
                pcm, sr = audio_queue.get_nowait()
                print("Frame: ", pcm)
                print("Sample rate: ", sr)
                frames.append(pcm)
                sample_rate = sr

            if frames:
                print("Frames is not empty!")
                st.session_state.audio_buffer = frames
                st.session_state.sample_rate = sample_rate
                st.session_state.processing = True
                st.rerun()          # trigger the processing block below

    # ── Processing block (runs synchronously on main thread) ─────────────────
    if st.session_state.processing:
        st.write("Processing audio with Whisper model…")
        st.write(f"Audio buffer has {len(st.session_state.audio_buffer)} frames at {st.session_state.sample_rate} Hz")
        
        time.sleep(5)  # small delay to ensure processing is complete
        # Drain again in case more frames arrived between rerun
        #frames, sample_rate = [], 16000
        #while not audio_queue.empty():
        #    pcm, sr = audio_queue.get_nowait()
        #    frames.append(pcm)
        #    sample_rate = sr

        with st.spinner("Transcribing your audio…"):
            # Load model takes time, so we load it here while showing a spinner. Cached loading means it will only happen once per model choice.
            model = load_whisper(st.session_state.last_model)
   
            transcript = transcribe_audio(st.session_state.audio_buffer, 
                                          st.session_state.sample_rate, model)

        st.session_state.transcript = transcript
        st.session_state.processing = False
        time.sleep(2)  # small delay to ensure state updates before rerun
        st.rerun()

    # ── Display transcript ────────────────────────────────────────────────────
    if st.session_state.transcript:
        st.markdown("### 📝 Transcript")
        st.write("Here's the transcribed text from your audio:", st.session_state.transcript)
        time.sleep(5)  # small delay to ensure transcript is set before rendering
        st.markdown(
            f'<div class="transcript-box">{st.session_state.transcript}</div>',
            unsafe_allow_html=True,
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                "⬇️ Download transcript",
                data=st.session_state.transcript,
                file_name="transcript.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with col_b:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.transcript = ""
                st.rerun()

else:
    # Stream not yet started
    st.info(
        "👆 Click **START** above to activate your microphone.\n\n"
        "On mobile, tap START and allow microphone access when prompted."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — settings & help
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    model_choice = st.selectbox(
        "Whisper model",
        ["tiny", "base", "small", "medium"],
        index=1,
        help="Larger models are more accurate but slower. 'base' is a good default.",
    )
    # Invalidate cached model if user changes selection
    if "last_model" not in st.session_state:
        st.session_state.last_model = model_choice
    if st.session_state.last_model != model_choice:
        st.session_state.last_model = model_choice
        # Clear the cache so the new model is loaded next time
        load_whisper.clear()
        st.rerun()

    st.divider()
    st.markdown("""
**How to use**
1. Click **START** to connect the mic
2. Click **🎤 Start Recording** to begin
3. Speak clearly into your device
4. Click **⏹️ Stop & Transcribe** when done
5. Your transcript appears below

**Tips for mobile**
- Use Chrome or Safari
- Allow microphone access when prompted
- Keep the screen on while recording
    """)

    st.divider()
    if not WHISPER_AVAILABLE:
        st.warning("Whisper not installed.\n\n`pip install openai-whisper`")
    else:
        st.success("Whisper ✅ ready")