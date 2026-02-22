import time
import torch
import numpy as np
from faster_whisper import WhisperModel
from config import SR_VAD, LANG, BEAM_SIZE, VAD_FILTER

def build_model(model_size: str):
    """faster_whisper 모델 로드"""
    print(f"[STT] Loading faster_whisper model: {model_size}")
    return WhisperModel(model_size, device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")

def transcribe_audio(model, audio_f32, lang=LANG, beam_size=BEAM_SIZE, vad_filter=VAD_FILTER, initial_prompt=None):
    t0 = time.time()
    segments, info = model.transcribe(audio_f32, language=lang, beam_size=beam_size, vad_filter=vad_filter, initial_prompt=initial_prompt)
    ms = (time.time() - t0) * 1000
    text = " ".join([seg.text for seg in segments]).strip()
    detected_lang = info.language if hasattr(info, 'language') else (lang or "")
    return text, detected_lang, ms
