import os
import time
import threading
import queue
import subprocess
import requests
import numpy as np
import sounddevice as sd
from pynput import keyboard
from faster_whisper import WhisperModel
from openai import OpenAI

import base64
import mss

from PIL import Image
import io

import win32gui

import sys

def reboot_script(listener):
    pressed_keys.clear()
    listener.stop()
    os.system("cls")
    time.sleep(0.05)
    python = sys.executable
    os.execl(python, python, *sys.argv)


SFX = {
    "listening": open("listening.mp3", "rb").read(),
    "screenshot": open("screenshot.mp3", "rb").read(),
}


# ============================================================
# Configuration
# ============================================================
# --- Whisper ---
MODEL_SIZE = "turbo"       # or "large-v3"
DEVICE = "cuda"            # "cuda" or "cpu"
COMPUTE_TYPE = "float16"   # "float16", "int8_float16", "int8"

SAMPLE_RATE = 16000 # whisper is trained at 16000 sample rate
CHANNELS = 1
DTYPE = "float32"

SCREENSHOT_QUALITY = 75

PUSH_TO_TALK_KEY = keyboard.Key.caps_lock

# Key state tracking for combos
pressed_keys = set()

CTRL_KEYS = {keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r}

VOICE_PITCH = 1

# --- OpenAI / ElevenLabs ---
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ELEVENLABS_API_KEY = os.environ["ELEVENLABS_API_KEY"]
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "JZLpE3AGwpKYZI2X65hN")

SYSTEM_PROMPT = (
    "You are a concise assistant. "
    "Answer in no more than 30 words unless I ask for a detailed answer. "
    "Do not use bullet points or lists. "
    "Do not use emojis or symbols which will not translate well into TTS."
)

from requests.adapters import HTTPAdapter

http = requests.Session()
http.headers.update({
    "xi-api-key": ELEVENLABS_API_KEY,
})

adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10)
http.mount("https://", adapter)
http.mount("http://", adapter)



def play_sfx_bytes(mp3_bytes: bytes):
    p = subprocess.Popen(
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-i", "pipe:0"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        p.stdin.write(mp3_bytes)
        p.stdin.close()   # CRITICAL: tells ffplay the stream is complete
    except Exception:
        pass


from datetime import datetime

SCREENSHOT_DIR = "screenshots"

def save_screenshot_bytes(jpeg_bytes: bytes) -> str:
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(SCREENSHOT_DIR, f"screenshot_{ts}.jpg")
    with open(path, "wb") as f:
        f.write(jpeg_bytes)
    return path


# ============================================================
# Clients
# ============================================================
client = OpenAI(api_key=OPENAI_API_KEY)

def gpt_chat(messages: list[dict]) -> str:
    resp = client.responses.create(
        model="gpt-4.1-mini",
        # model="gpt-4-turbo",
        input=messages,
        max_output_tokens=120,
    )
    return resp.output_text.strip()

def elevenlabs_stream_play(text: str) -> None:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream"

    headers = {
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }

    payload = {
        "text": text,
        "model_id": "eleven_flash_v2_5",
        "voice_settings": {
            "stability": 0.65,
            "similarity_boost": 0,
            "speed": 1.2,
        },
    }

    query = {
        "output_format": "mp3_22050_32",
        "optimize_streaming_latency": 3,
    }

    r = http.post(url, headers=headers, params=query, json=payload, stream=True, timeout=60)

    if r.status_code >= 400:
        print("ElevenLabs error:", r.status_code)
        print(r.text)
        r.raise_for_status()

    p = subprocess.Popen(
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-i", "pipe:0"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        for chunk in r.iter_content(chunk_size=4096):
            if chunk:
                p.stdin.write(chunk)
        p.stdin.close()
        p.wait()
    finally:
        try:
            p.kill()
        except Exception:
            pass

# ============================================================
# Whisper model loading (with status)
# ============================================================
print("Loading Whisper model...")
load_start = time.perf_counter()
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
load_end = time.perf_counter()
print(f"Whisper model loaded in {load_end - load_start:.2f} seconds.")

# ============================================================
# Recording state
# ============================================================
audio_q: queue.Queue[np.ndarray] = queue.Queue()
recording_lock = threading.Lock()
is_recording = False
frames: list[np.ndarray] = []
stream: sd.InputStream | None = None

# Prevent recording while we’re speaking (optional but helpful)
playback_lock = threading.Lock()
is_playing_tts = False

# Conversation state
messages = [{"role": "system", "content": SYSTEM_PROMPT}]

def audio_callback(indata, frame_count, time_info, status):
    if status:
        # If you want to debug dropouts/overruns, print(status)
        pass

    with recording_lock:
        if not is_recording:
            return

    # Avoid capturing mic while TTS is playing (best-effort)
    with playback_lock:
        if is_playing_tts:
            return

    audio_q.put(indata.copy())

def start_stream_if_needed():
    global stream
    if stream is None:
        print("Initializing audio input stream...")
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=audio_callback,
        )
        stream.start()
        print("Audio input stream ready.")

def drain_queue_into_frames():
    while True:
        try:
            chunk = audio_q.get_nowait()
        except queue.Empty:
            break
        frames.append(chunk)

def begin_recording():
    global is_recording, frames
    start_stream_if_needed()

    with playback_lock:
        if is_playing_tts:
            return

    with recording_lock:
        if is_recording:
            return
        is_recording = True

    play_sfx_bytes(SFX["listening"])


    frames = []
    drain_queue_into_frames()
    print("Recording started.")
    
def transcribe_audio_array(audio: np.ndarray) -> tuple[str, str, float]:
    """
    audio: 1D float32 array at SAMPLE_RATE (16000)
    Returns (text, language, language_probability).
    """
    # Ensure float32, mono, contiguous
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    segments, info = model.transcribe(
        audio,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=250),
    )
    text = "".join(seg.text for seg in segments).strip()
    return text, info.language, float(info.language_probability)


def end_recording_and_transcribe_then_chat():
    global is_recording
    with recording_lock:
        if not is_recording:
            return
        is_recording = False

    drain_queue_into_frames()

    if not frames:
        print("Recording stopped. No audio captured.")
        return

    print("Recording stopped. Transcribing...")

    audio = np.concatenate(frames, axis=0).reshape(-1)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio /= peak

    t0 = time.perf_counter()
    try:
        text, lang, prob = transcribe_audio_array(audio)
    except Exception as e:
        print(f"Transcription failed: {e}")
        return
    t1 = time.perf_counter()


    print(f"Transcription finished in {t1 - t0:.2f} seconds.")
    if not text:
        print("(No speech detected)")
        return

    print(f"[{lang} {prob:.2f}] {text}\n")

    # IMPORTANT: do GPT + TTS in a background thread so the keyboard listener stays responsive
    threading.Thread(target=handle_user_text, args=(text,), daemon=True).start()
    
    
def capture_active_window_jpeg_bytes(
    quality: int = SCREENSHOT_QUALITY,
    # max_width: int = 1280,
    # max_height: int = 720,    
    max_width: int = 3000,
    max_height: int = 3000,
) -> bytes:
    """
    Captures the currently active window only.
    If the image exceeds max_width or max_height,
    it is downscaled proportionally to fit within those bounds.
    """

    hwnd = win32gui.GetForegroundWindow()
    if not hwnd:
        raise RuntimeError("No active window detected")

    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width = right - left
    height = bottom - top

    if width <= 0 or height <= 0:
        raise RuntimeError("Invalid window dimensions")

    with mss.mss() as sct:
        monitor = {
            "left": left,
            "top": top,
            "width": width,
            "height": height,
        }
        shot = sct.grab(monitor)

        img = Image.frombytes("RGB", shot.size, shot.rgb)

        # ---- Conditional downscale ----
        if img.width > max_width or img.height > max_height:
            scale = min(
                max_width / img.width,
                max_height / img.height,
            )
            new_size = (
                int(img.width * scale),
                int(img.height * scale),
            )
            img = img.resize(new_size, Image.BILINEAR)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=SCREENSHOT_QUALITY, optimize=True)
        return buf.getvalue()

def jpeg_bytes_to_data_url(jpeg_bytes: bytes) -> str:
    b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def handle_screenshot():
    """
    Capture screenshot and send it to GPT as a vision message.
    """
    global messages, is_playing_tts

    # Capture screenshot
    try:
        jpeg_bytes = capture_active_window_jpeg_bytes(quality=92)
        saved_path = save_screenshot_bytes(jpeg_bytes)
        print(f"Saved screenshot: {saved_path}")
        data_url = jpeg_bytes_to_data_url(jpeg_bytes)
    except Exception as e:
        print(f"Screenshot capture failed: {e}")
        return


    # Build a vision turn (text + image in one user message)
    vision_prompt = (
        "Describe what you see in exactly 5 words. "
    )

    user_msg = {
        "role": "user",
        "content": [
            {"type": "input_text", "text": vision_prompt},
            {
                "type": "input_image",
                "image_url": data_url,
                "detail": "low",  # faster/cheaper; change to "high" if you need small text detail
            },
        ],
    }

    messages.append(user_msg)

    try:
        reply = gpt_chat(messages)
    except Exception as e:
        print(f"GPT request failed (screenshot): {e}")
        return

    print(f"GPT (screenshot): {reply}\n")
    messages.append({"role": "assistant", "content": reply})
    
    with open(saved_path.replace(".jpg", ".txt"), "w", encoding="utf-8") as f:
        f.write(reply)


    with playback_lock:
        is_playing_tts = True
    try:
        elevenlabs_stream_play(reply)
    finally:
        with playback_lock:
            is_playing_tts = False

def handle_user_text(user_text: str):
    """
    Send text to GPT, print reply, run TTS, play audio.
    """
    global messages, is_playing_tts

    messages.append({"role": "user", "content": user_text})

    try:
        reply = gpt_chat(messages)
    except Exception as e:
        print(f"GPT request failed: {e}")
        return

    print(f"GPT: {reply}\n")
    messages.append({"role": "assistant", "content": reply})

    with playback_lock:
        is_playing_tts = True
    try:
        elevenlabs_stream_play(reply)
    finally:
        with playback_lock:
            is_playing_tts = False


def on_press(key, listener):
    pressed_keys.add(key)

    ctrl_held = any(ctrl in pressed_keys for ctrl in CTRL_KEYS)

    if ctrl_held:
        ch = getattr(key, "char", None)
        if ch in ("r", "R", "\x12"):
            reboot_script(listener)
            return False


    if key == PUSH_TO_TALK_KEY and not ctrl_held:
        begin_recording()

    if key == PUSH_TO_TALK_KEY and ctrl_held:
        play_sfx_bytes(SFX["screenshot"])
        threading.Thread(target=handle_screenshot, daemon=True).start()


def on_release(key):
    pressed_keys.discard(key)

    ctrl_held = any(ctrl in pressed_keys for ctrl in CTRL_KEYS)

    # Only stop recording if it was started as push-to-talk
    if key == PUSH_TO_TALK_KEY and not ctrl_held:
        end_recording_and_transcribe_then_chat()



def main():
    print("Voice Chat: Hold CAPS LOCK to talk. Release to send. Press CTRL + CAPS LOCK to send a screenshot of currently active window. You can reboot with CTRL + R\n")
    start_stream_if_needed()

    def _on_press(key):
        on_press(key, listener)

    def _on_release(key):
        on_release(key)

    with keyboard.Listener(on_press=_on_press, on_release=_on_release) as listener:
        listener.join()


if __name__ == "__main__":
    main()
