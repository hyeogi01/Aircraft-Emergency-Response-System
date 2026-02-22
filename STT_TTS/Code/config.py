import os

# ===== 경로 설정 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOICES_DIR = os.path.join(BASE_DIR, "../voices")

# ===== STT =====
DEVICE_INDEX = "hw:2,0"            # USB 마이크 인덱스 (python -m sounddevice 로 확인)
SR_IN = 48000                # 마이크 기본 샘플레이트 (48kHz)
SR_VAD = 16000                # VAD/Whisper 입력 샘플레이트
BLOCK_MS = 30                 # VAD 허용: 10/20/30 ms
VAD_AGGR = 2                  # 0~3 (높을수록 침묵을 공격적으로 감지)
LANG = None                   # None = 자동감지
MODEL_SIZE = "medium"         # tiny/base/small/medium/large-v3
BEAM_SIZE = 5
VAD_FILTER = False

MIN_SPEECH_SEC = 0.6          # 연속 발화 최소 시간
MIN_SILENCE_SEC = 0.5         # 연속 무음 시간
MAX_SEGMENT_SEC = 20          # 최대 세그먼트 길이

# ===== 파생값 =====
INT16_MAX = 32768.0
FRAME_IN = int(SR_IN * (BLOCK_MS / 1000.0))
FRAME_VAD = int(SR_VAD * (BLOCK_MS / 1000.0))
BYTES_VAD = FRAME_VAD * 2

MIN_SPEECH_FRAMES = int(MIN_SPEECH_SEC * 1000 / BLOCK_MS)
MIN_SILENCE_FRAMES = int(MIN_SILENCE_SEC * 1000 / BLOCK_MS)
MAX_FRAMES = int(MAX_SEGMENT_SEC * 1000 / BLOCK_MS)

# ===== INITIAL PROMPT =====
INITIAL_PROMPT = (
    "비행, 중, 좌측, 엔진에서, 연기가, 감지되어, 엔진, 화재가, "
    "의심되는, 경우, 어떤, 절차를, 적용해야, 해, "
    "현재, 연료, 불균형, FUEL IMBALANCE, 발생했어, "
    "대응, 절차를, 알려줘, "
    "양쪽, 엔진, 블리드가, 다, 죽었는데, APU, 블리드로, 공급이, 가능한가"
)

# ===== KEY WORDS =====
RAG_KEYWORDS = ["엔진", "화재", "절차", "확인", "연료", "불균형", "블리드", "고장", "APU", "에이피유", "에어", "공급"]
CHECK_KEYWORDS = ["절차", "알려줘", "확인", "다음", "체크"]