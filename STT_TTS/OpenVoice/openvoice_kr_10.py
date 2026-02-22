import os
import torch
import time
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS

# 경로 설정
ckpt_base = "checkpoints_v2/base_speakers/ses"
ckpt_converter = "checkpoints_v2/converter"
output_dir = "../voices/TTS_results/openvoice_output"

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(output_dir, exist_ok=True)

# Tone Color Converter
tone_color_converter = ToneColorConverter(f"{ckpt_converter}/config.json", device=device)
tone_color_converter.load_ckpt(f"{ckpt_converter}/checkpoint.pth")

# 참조 화자 (클론할 목소리)
reference_speaker = "resources/test3.wav"
target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=True)

# 합성할 100개 문장
texts = [
    "엔진에서 일어난 화재입니다.",
    "기체의 왼쪽 날개가 손상되었습니다.",
    "산소 마스크를 착용하십시오.",
    "승객 대피 절차를 시작하세요.",
    "비상 착륙을 준비하십시오.",
    "기내 압력이 떨어지고 있습니다.",
    "연료 누출이 감지되었습니다.",
    "조종석 화재 경보가 울렸습니다.",
    "우측 엔진이 정지했습니다.",
    "관제탑에 비상 상황을 알리십시오.",

    "오늘의 날씨를 알려드리겠습니다.",
    "현재 시각은 오후 세 시입니다.",
    "서울의 기온은 영상 스물도입니다.",
    "내일은 비가 내릴 예정입니다.",
    "오늘 하루도 고생 많으셨습니다.",
    "좋은 하루 되세요.",
    "안전벨트를 꼭 착용하시기 바랍니다.",
    "전원 꺼짐을 확인해 주십시오.",
    "비행기는 곧 착륙할 예정입니다.",
    "기내에서는 금연입니다.",

    "좌석 등받이를 세워 주십시오.",
    "휴대폰 전원을 꺼 주시기 바랍니다.",
    "비상구는 앞뒤에 위치해 있습니다.",
    "승무원의 안내에 따라 주십시오.",
    "기내 온도는 섭씨 이십이도입니다.",
    "안전 브리핑을 시작하겠습니다.",
    "기내 조명을 낮추겠습니다.",
    "착륙 준비를 완료해 주십시오.",
    "기내식은 곧 제공됩니다.",
    "오늘도 저희 항공사를 이용해 주셔서 감사합니다.",

    "승무원은 비상 장비를 점검하십시오.",
    "승객 여러분은 침착하게 행동해 주십시오.",
    "비상용 산소 마스크는 좌석 위에 있습니다.",
    "구명조끼는 좌석 아래에 있습니다.",
    "구명정은 양쪽 날개 옆에 있습니다.",
    "긴급 탈출구는 네 개가 있습니다.",
    "비행기는 현재 순항 중입니다.",
    "승무원 호출 버튼은 머리 위에 있습니다.",
    "객실 내 압력이 안정되었습니다.",
    "비상 조명은 자동으로 켜집니다.",

    "오늘의 주요 뉴스입니다.",
    "세계 경제가 회복세를 보이고 있습니다.",
    "주식 시장은 소폭 상승했습니다.",
    "달러 환율이 하락했습니다.",
    "기술 발전이 빠르게 이루어지고 있습니다.",
    "인공지능이 다양한 분야에서 활용되고 있습니다.",
    "환경 보호의 중요성이 커지고 있습니다.",
    "기후 변화가 심각한 문제로 떠오르고 있습니다.",
    "온실가스 배출이 증가하고 있습니다.",
    "지구 평균 기온이 상승하고 있습니다.",

    "오늘 아침에 비가 내렸습니다.",
    "점심 메뉴는 김치찌개입니다.",
    "저녁에 친구를 만날 예정입니다.",
    "주말에 영화를 볼 계획입니다.",
    "내일 아침 일찍 출근해야 합니다.",
    "버스를 타고 학교에 갑니다.",
    "지하철은 붐비고 있습니다.",
    "택시는 빠르지만 비쌉니다.",
    "자동차가 도로에 많습니다.",
    "자전거를 타고 공원에 갔습니다.",

    "컴퓨터를 켜고 인터넷을 연결했습니다.",
    "휴대폰 배터리가 부족합니다.",
    "충전기를 꽂아야 합니다.",
    "무선 이어폰을 사용하고 있습니다.",
    "텔레비전 소리를 줄여 주세요.",
    "냉장고에 우유가 있습니다.",
    "커피를 마시고 싶습니다.",
    "책을 읽는 것이 즐겁습니다.",
    "음악을 크게 틀지 마세요.",
    "영화를 재미있게 보았습니다.",

    "학교에 지각했습니다.",
    "시험 준비를 열심히 하고 있습니다.",
    "숙제를 제출해야 합니다.",
    "도서관에서 공부하고 있습니다.",
    "선생님께 질문을 했습니다.",
    "친구와 운동장에서 놀았습니다.",
    "점심시간에 급식을 먹었습니다.",
    "방과 후에 동아리 활동이 있습니다.",
    "교실에 책상이 많습니다.",
    "칠판에 글씨를 썼습니다.",

    "오늘의 회의는 오후 두 시에 시작합니다.",
    "프로젝트 진행 상황을 보고합니다.",
    "일정이 지연되고 있습니다.",
    "예산이 부족합니다.",
    "인력을 충원해야 합니다.",
    "고객의 만족도가 중요합니다.",
    "품질 관리를 강화해야 합니다.",
    "경쟁사가 새로운 제품을 출시했습니다.",
    "마케팅 전략을 수정해야 합니다.",
    "매출이 증가하고 있습니다.",

    "이 비행기는 최신 기종입니다.",
    "비행 시간은 세 시간입니다.",
    "이륙 준비를 하고 있습니다.",
    "목적지는 부산입니다.",
    "비행 고도는 만 미터입니다.",
    "승객 수는 백오십 명입니다.",
    "기내 서비스가 제공됩니다.",
    "기장님이 안내 방송을 합니다.",
    "기체 흔들림이 예상됩니다.",
    "안전 운항을 위해 최선을 다하겠습니다."
]

# TTS (한국어 모델)
model = TTS(language="KR", device=device)
speaker_ids = model.hps.data.spk2id

for i, text in enumerate(texts, start=1):
    for speaker_key, speaker_id in speaker_ids.items():
        speaker_key = speaker_key.lower().replace("_", "-")

        start_time = time.time()   # 시작 시간 기록

        # 기본 음성 생성 (임시)
        src_path = f"{output_dir}/tmp_{speaker_key}_{i}.wav"
        model.tts_to_file(text, speaker_id, src_path, speed=1.0)

        # 음색 변환 (최종 파일)
        converted_path = f"{output_dir}/openvoice_{i}.wav"
        encode_message = "@MyShell"  # watermark
        tone_color_converter.convert(
            audio_src_path=src_path,
            src_se=torch.load(f"{ckpt_base}/{speaker_key}.pth", map_location=device),
            tgt_se=target_se,
            output_path=converted_path,
            message=encode_message,
        )

        end_time = time.time()
        elapsed = end_time - start_time

        print(f"✅ 변환된 파일: {converted_path}")
        print(f"⏱ 실행 시간: {elapsed:.2f}초")
