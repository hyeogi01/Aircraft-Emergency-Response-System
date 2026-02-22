import numpy as np
import os
import soundfile as sf

VOICES_DIR = "../voices"

def get_path(filename):
    if not filename.lower().endswith(".wav"):
        filename += ".wav"
    return os.path.join(VOICES_DIR, filename)

def load_wav_to_float32(filename):
    path = get_path(filename)
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio, sr
