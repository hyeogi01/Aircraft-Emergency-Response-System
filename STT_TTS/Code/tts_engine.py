import os
import time
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS
from datetime import datetime

session_timestamp = None
file_counter = 1

# --- 초기화 부분 ---
ckpt_base = "../OpenVoice/checkpoints_v2/base_speakers/ses"
ckpt_converter = "../OpenVoice/checkpoints_v2/converter"
output_dir = "../voices/TTS_results"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[TTS] torch.cuda.is_available()={torch.cuda.is_available()}")
print(f"[TTS] 선택된 device={device}")
if device == "cuda":
    print(f"[TTS] GPU name: {torch.cuda.get_device_name(0)}")
    print(f"[TTS] GPU 메모리 (할당/총량): {torch.cuda.memory_allocated(0)/1024**2:.1f} MB / {torch.cuda.get_device_properties(0).total_memory/1024**2:.1f} MB")
os.makedirs(output_dir, exist_ok=True)

# ToneColorConverter 로드
tone_color_converter = ToneColorConverter(f"{ckpt_converter}/config.json", device=device)
tone_color_converter.load_ckpt(f"{ckpt_converter}/checkpoint.pth")
print(f"[TTS] ToneColorConverter device={tone_color_converter.device}")

# 참조 화자 (목소리 클론 대상)
reference_speaker = "../voices/test3.wav"
target_se, _ = se_extractor.get_se(reference_speaker, tone_color_converter, vad=True)

# Melo TTS 모델
tts_model = TTS(language="EN", device=device)
print(f"[TTS] Melo TTS device={tts_model.device}")
speaker_ids = tts_model.hps.data.spk2id


def synthesize_and_convert(text: str):
    """
    텍스트를 받아 음성 합성 + 톤 변환을 수행하고 파일 경로 반환
    """
    global session_timestamp, file_counter

    if not text.strip():
        print("[TTS] 입력 텍스트 없음 (skip)")
        return None
    # 첫 번째 실행 시점 타임스탬프 고정
    if session_timestamp is None:
        session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for speaker_key, speaker_id in speaker_ids.items():
        speaker_key = speaker_key.lower().replace("_", "-")

        print(f"[DEBUG] TTS 시작 --> speaker={speaker_key}, text={text}")

        start_time = time.time()

        try:
            # 1단계: 원본 합성 (임시 wav)
            src_path = f"{output_dir}/tmp_{speaker_key}_{file_counter}.wav"
            print(f"[DEBUG] tts_model.tts_to_file 호출: {src_path}")
            tts_model.tts_to_file(text, speaker_id, src_path, speed=1.0)
            print(f"[DEBUG] 원본 합성 완료 --> {src_path}")

            # 2단계: 톤 컬러 변환 (최종 wav)
            # converted_filename = f"tts_output_{session_timestamp}_{file_counter}.wav"
            converted_filename = f"emergency_{file_counter}"
            converted_path = os.path.join(output_dir, converted_filename)

            print(f"[DEBUG] tone_color_converter.convert 호출: {converted_path}")
            encode_message = "@MyShell"  # watermark
            tone_color_converter.convert(
                audio_src_path=src_path,
                src_se=torch.load(f"{ckpt_base}/{speaker_key}.pth", map_location=device),
                tgt_se=target_se,
                output_path=converted_path,
                message=encode_message,
            )
            print(f"[DEBUG] 톤 컬러 변환 완료 --> {converted_path}")

            elapsed = time.time() - start_time
            print(f"✅ 변환된 파일: {converted_path}")
            print(f"⏱ 실행 시간: {elapsed:.2f}초")

            file_counter += 1  # 다음 호출 대비 증가
            return converted_path  # 한 개만 반환

        except Exception as e:
            print(f"[ERROR] TTS 변환 중 오류 발생: {e}")
            return None
