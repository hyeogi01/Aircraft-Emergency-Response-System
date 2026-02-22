import sys, queue
import numpy as np
import sounddevice as sd
import webrtcvad
from scipy.signal import resample_poly
from config import *

seg_q = queue.Queue()

class VADMicStream:
    def __init__(self):
        self.raw_q = queue.Queue()
        self.vad = webrtcvad.Vad(VAD_AGGR)
        self.resamp_buf = bytearray()

    def audio_cb(self, indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        self.raw_q.put(bytes(indata))

    def rs_48k_to_16k(self, int16_bytes_48k):
        x = np.frombuffer(int16_bytes_48k, dtype=np.int16).astype(np.float32)
        y = resample_poly(x, 1, 3)  # 48000 → 16000
        y = np.clip(y, -32768, 32767).astype(np.int16)
        return y.tobytes()

    def run(self):
        in_speech = False
        voiced_buf = bytearray()
        speech_frames = 0
        silence_frames = 0

        stream = sd.RawInputStream(
            samplerate=SR_IN, channels=1, dtype="int16",
            blocksize=FRAME_IN, callback=self.audio_cb
        )
        with stream:
            while True:
                b = self.raw_q.get()
                if not b:
                    continue
                self.resamp_buf.extend(self.rs_48k_to_16k(b))

                while len(self.resamp_buf) >= BYTES_VAD:
                    frame16 = bytes(self.resamp_buf[:BYTES_VAD])
                    del self.resamp_buf[:BYTES_VAD]
                    is_speech = self.vad.is_speech(frame16, SR_VAD)

                    if is_speech:
                        voiced_buf.extend(frame16)
                        speech_frames += 1
                        silence_frames = 0
                        if not in_speech and speech_frames >= MIN_SPEECH_FRAMES:
                            in_speech = True
                    else:
                        if in_speech:
                            silence_frames += 1
                            voiced_buf.extend(frame16)
                            if silence_frames >= MIN_SILENCE_FRAMES or len(voiced_buf) >= MAX_FRAMES * BYTES_VAD:
                                seg_q.put(bytes(voiced_buf))
                                voiced_buf.clear()
                                in_speech = False
                                speech_frames = 0
                                silence_frames = 0
                        else:
                            speech_frames = 0

def run_vad_stream():
    VADMicStream().run()
