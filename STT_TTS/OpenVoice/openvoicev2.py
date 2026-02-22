import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter, BaseSpeakerTTS

# 체크포인트 경로
ckpt_base = "checkpoints_v2/base_speakers/ses"   # 한국어 kr.pth 있는 경로
ckpt_converter = "checkpoints_v2/converter"
output_dir = "outputs"

# CUDA 또는 CPU 선택
device = "cuda" if torch.cuda.is_available() else "cpu"

# Base Speaker TTS (한국어 kr.pth 사용)
# 올바른 코드
base_speaker_tts = BaseSpeakerTTS(os.path.join(ckpt_base, "config.json"), device=device)
base_speaker_tts.load_ckpt(os.path.join(ckpt_base, "kr.pth"))

# Tone Color Converter
tone_color_converter = ToneColorConverter(os.path.join(ckpt_converter, "config.json"), device=device)
tone_color_converter.load_ckpt(os.path.join(ckpt_converter, "checkpoint.pth"))

os.makedirs(output_dir, exist_ok=True)

# 테스트 문장
text = "안녕하세요, 이것은 OpenVoice V2의 한국어 음성 합성 예제입니다."

# 1단계: 기본 한국어 음성 생성
src_path = os.path.join(output_dir, "tmp.wav")
base_speaker_tts.tts(text, src_path, speaker="KR", language="KR")

# 2단계: 톤/컬러 보정 (선택사항: 다른 화자 성향에 맞추고 싶을 때)
# reference_speaker = "samples/ref_kr.wav"
# target_se, _ = se_extractor.get_se(reference_speaker, tone_color_converter, device=device)
# converted_path = os.path.join(output_dir, "final.wav")
# tone_color_converter.convert(
#     audio_src_path=src_path,
#     src_se=target_se,
#     tgt_se=target_se,
#     output_path=converted_path
# )
# print("최종 결과:", converted_path)

print("생성된 음성:", src_path)
