import os
import torch
import time
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS

# 경로 설정
ckpt_base = "checkpoints_v2/base_speakers/ses"
ckpt_converter = "checkpoints_v2/converter"
# output_dir = "outputs_v2"
output_dir = "../voices/TTS_results/openvoice_output"

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(output_dir, exist_ok=True)

# Tone Color Converter
tone_color_converter = ToneColorConverter(f"{ckpt_converter}/config.json", device=device)
tone_color_converter.load_ckpt(f"{ckpt_converter}/checkpoint.pth")

# 참조 화자 (클론할 목소리)
reference_speaker = "resources/test3.wav"
target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=True)

# 합성할 한국어 문장
text = "조준기 사랑해"

# TTS (한국어 모델)
model = TTS(language="KR", device=device)
speaker_ids = model.hps.data.spk2id

for speaker_key, speaker_id in speaker_ids.items():
    speaker_key = speaker_key.lower().replace("_", "-")

    start_time = time.time()   # 시작 시간 기록

    # 기본 음성 생성
    src_path = f"{output_dir}/tmp_{speaker_key}.wav"
    model.tts_to_file(text, speaker_id, src_path, speed=1.0)

    # 음색 변환
    converted_path = f"{output_dir}/openvoice_.wav"
    encode_message = "@MyShell"  # watermark
    tone_color_converter.convert(
        audio_src_path=src_path,
        src_se=torch.load(f"{ckpt_base}/{speaker_key}.pth", map_location=device),
        tgt_se=target_se,
        output_path=converted_path,
        message=encode_message,
    )

    end_time = time.time()     # 끝 시간 기록
    elapsed = end_time - start_time

    print(f"✅ 변환된 파일: {converted_path}")
    print(f"⏱ 실행 시간: {elapsed:.2f}초")
