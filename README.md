# Simple-Desktop-AI-Assistant
Uses local speech recognition, GPT for thinking, ElevenLabs for output. Can have conversations and analyse screenshot of current window with a keypress. Handy for taking notes or asking random questions without needing to pull up the GPT chat interface.

Created using GPT 5.2.

## Features

- Push-to-talk voice input using **Caps Lock**
- Local speech recognition via **faster-whisper**
- GPU acceleration (CUDA / FP16 supported)
- GPT-based conversational responses
- Streaming text-to-speech via **ElevenLabs**
- Screenshot capture of the active window (**Ctrl + Caps Lock**)
- Automatic screenshot downscaling and JPEG compression
- Saved screenshots with paired text descriptions
- Non-blocking background processing (keyboard remains responsive)

## Controls

| Action | Shortcut |
|------|---------|
| Record voice | **Caps Lock** |
| Capture screenshot + vision analysis | **Ctrl + Caps Lock** |
| Restart script | **Ctrl + R** |

## How It Works

### 1. Audio Input
- Microphone audio is captured only while Caps Lock is held
- Audio is buffered and normalized
- **faster-whisper** performs local transcription

### 2. GPT Interaction
- Transcribed text is appended to the conversation history
- GPT generates a concise response
- Responses are capped for TTS clarity

### 3. Text-to-Speech
- ElevenLabs streams MP3 audio in real time
- Audio is played immediately via `ffplay`
- Mic capture pauses during playback to prevent feedback

### 4. Screenshot Vision Mode
- Captures only the currently active window
- Automatically downscales oversized windows
- Compresses to optimized JPEG
- Sends image + prompt to GPT-Vision
- Saves:
  - `.jpg` screenshot
  - `.txt` description alongside it

## Requirements

### Hardware
- Windows (uses `win32gui`)
- Microphone
- Optional: NVIDIA GPU for CUDA acceleration

### Software
- Python 3.10+
- FFmpeg (for `ffplay`)
- CUDA (optional but recommended)

### Environment Variables
```
OPENAI_API_KEY=your_openai_key
ELEVENLABS_API_KEY=your_elevenlabs_key
ELEVENLABS_VOICE_ID=your_voice_id   # optional
```

### Dependencies

```pip install faster-whisper sounddevice pynput numpy pillow mss pywin32 openai requests```

### Configuration
```
MODEL_SIZE = "turbo"          # or "large-v3"
DEVICE = "cuda"               # "cuda" or "cpu"
COMPUTE_TYPE = "float16"      # float16 / int8 / int8_float16
SCREENSHOT_QUALITY = 75
PUSH_TO_TALK_KEY = keyboard.Key.caps_lock
```

