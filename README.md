## 프로젝트명
- 항공기 비정상 상황 대응 AI 시스템
## 추진 개요
- 실제 운항 환경에서 조종사 인지 부담을 줄일 수 있는 AI 지원체계 구축 필요성 대두
## 구현 목표
- 조종실 안전관리 CV 모델
- 대화형 절차 안내 LLM
- 실시간 모니터링 시스템

→ 비정상 상황에서 조종사의 인지 부담을 줄이고, 실시간 절차를 안내하는 AI 보조 시스템 구현

## 논문 분석
- LoRE: 질의응답 향상을 위한 로짓 기반 앙상블 검색기
- 자폐 화자의 음성 인식을 위한 파인튜닝 기법

## 시스템 구성도
<img width="1179" height="568" alt="image" src="https://github.com/user-attachments/assets/2b7444dc-64cf-4dfb-9939-e5eec755a89f" />

## 구현 기술
1. CV
    - YOLOv11, Face Mesh, Face recognition, DehazeFormer 등 사용
    - 인가된 사람 감지 (YOLOv11, Face recognition)
      - 임베딩 기반 코사인 유사도를 통해 인가자를 판단
    - 조종사 표정 감지 (Face Mesh + LSTM) → 시계열 분석을 토대로 조종사 표정 감지(졸음,실신 감지)
    - 안개/연기 제거 (DehazeFormer)
2. STT/TTS
    <img width="1101" height="598" alt="image" src="https://github.com/user-attachments/assets/add5cfcd-df23-4846-910b-dbbf20162c5e" />

    <img width="1109" height="593" alt="image" src="https://github.com/user-attachments/assets/7b6aa2fb-038b-4fa8-9925-0f31ec2c1abf" />

    - Faster-Whisper 사용 (파라미터: Medium, beam size=5) → WER, CER, RTF를 토대로 선정
    - LoRA와 비교했을 때, 최적 Parameter + Initial Prompt가 성능이 높아 채택
    - 단어 추출은 KIWIPIEPY를 이용, 형태소 분석 기법 사용
    - TTS는 OpenVoice2 채택
4. RAG
    <img width="1212" height="596" alt="image" src="https://github.com/user-attachments/assets/a3e0597d-6bba-48cc-9b47-52d16d2b7f16" />

    - 앙상블 리트리버 도입 (기존 리트리버에 비해 검색도 향상)
    - BERT Score (F1) - 생성 답변과 정답의 ‘의미적 유사도’ 비교
    - 민항기 조종사 15명 대상 모델 비교(Ours, Mistral-7B, Gemini-2.5) 설문조사 진행
5. 웹 상 서비스 구현

## 기대 효과
<img width="1114" height="574" alt="image" src="https://github.com/user-attachments/assets/7f151cc5-1379-4560-a8ff-1169f1b08aa2" />

## 제한점 및 개선기회
<img width="1115" height="566" alt="image" src="https://github.com/user-attachments/assets/1bfe63ca-53ef-430b-b4af-42bf88c933ee" />
