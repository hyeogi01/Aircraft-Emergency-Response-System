import socket
import threading
import queue
from stt_engine import build_model, transcribe_audio
from tts_engine import synthesize_and_convert
from vad_stream import run_vad_stream, seg_q
from pos_utils import extract_nouns
from audio_utils import load_wav_to_float32
import numpy as np
import sys
import time
import websocket
from pydub import AudioSegment
from pydub.playback import play
import glob
from config import *
# from tts_engine import tts_worker

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

stt_result_q = queue.Queue()

# ws = websocket.WebSocket()
# ws.connect("ws://localhost:8000/ws/stt")
# print("[CHATLOG] WebSocket 연결 완료 → ws://localhost:8000/ws/stt")
ws = None

CHECK_DIR = "../voices/TTS_results"
checklist_files = sorted(glob.glob(os.path.join(CHECK_DIR, "emergency_*")))
check_step = 1

def play_next_checklist():
    global check_step
    if check_step >= len(checklist_files):
        print("[CHECKLIST] 모든 항목 완료")
        return None

    wav_file = checklist_files[check_step]
    print(f"[CHECKLIST] 재생: {wav_file}")
    sound = AudioSegment.from_file(wav_file, format="wav")
    play(sound)

    idx = check_step
    check_step += 1

    if ws:
        try:
            ws.send(f"check,{idx-1}")
        except Exception as e:
            print(f"[CHECKLIST] WebSocket 전송 실패: {e}")

    return idx

def stt_worker(rag_sock):
    print("[STT] 모델 로딩 중...")
    model = build_model(MODEL_SIZE)
    print("[STT] 모델 로딩 완료 ✅")

    # 👉 디바이스 및 연산 타입 확인 출력
    try:
        ct2_model = model.model  # WhisperModel 안의 Translator 객체
        print(f"[STT] Device: {ct2_model.device}")
        print(f"[STT] Device index: {ct2_model.device_index}")
        print(f"[STT] Compute type: {ct2_model.compute_type}")
    except Exception as e:
        print("[STT] 디바이스 정보를 가져올 수 없습니다:", e)

    print("🎤 마이크 입력을 시작하세요!")
    threading.Thread(target=run_vad_stream, daemon=True).start()

    print("🎙️ 준비 완료! 말씀하시면 인식이 시작됩니다...")

    while True:
        pcm16_bytes = seg_q.get()
        audio_f32 = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        text, lang, ms = transcribe_audio(model, audio_f32, LANG, BEAM_SIZE, VAD_FILTER, INITIAL_PROMPT)
        if text:
            nouns = extract_nouns(text)
            print(f"[STT/{lang}] {text} ({ms:.0f}ms) | Nouns: {nouns}")

            if any(kw in nouns for kw in RAG_KEYWORDS):
                # STT -> TTS 할 때 사용
                # stt_result_q.put(text)
                # print(f"[STT] Trigger 단어 감지 → 큐에 추가: {text}")
                # STT -> RAG 할 때 사용
                rag_sock.sendall((text + "\n").encode("utf-8"))
                ws.send(f"user,{text}")
            elif any(kw in nouns for kw in CHECK_KEYWORDS):
                print("[STT] CHECK Trigger 감지 -> 체크리스트 진행")
                play_next_checklist()
                # sound = AudioSegment.from_file("../voices/test3.wav", format="wav")
                # play(sound)
                # 음성 재생
                pass
            else:
                print("[STT] Trigger 단어 없음 → 큐에 추가 안 함")
        else:
            print("[STT] (무음/인식없음)")

def tts_worker(result_q):
    """큐에서 텍스트 꺼내서 TTS 합성"""
    while True:
        text = result_q.get()
        if text:
            if text.strip() in {"Answer generated successfully.", "답변 생성 완료 했습니다."}:
                ws.send(f"bot,CHECKLIST 생성 완료 했습니다.")
                print(f"[TTS] 건너뜀 (완료 신호): {text}")
                continue
            synthesize_and_convert(text)
            print(f"[TTS] 변환 완료: {text}")
            if result_q.empty():
                try:
                    ws.send(f"bot,답변 생성 완료 했습니다.")
                    print("[TTS] 마지막 문장 처리 완료 → ChatLog 알림 전송")
                except Exception as e:
                    print(f"[TTS] WebSocket 전송 실패: {e}")

def rag_listener(rag_sock, stt_result_q):
    """RAG 서버에서 답변 수신 -> 큐에 추가"""
    while True:
        try:
            data = rag_sock.recv(4096)
            if not data:
                print("[RAG] 연결 종료")
                break
            answer = data.decode("utf-8").strip()
            if answer:
                if answer.strip() in {"Answer generated successfully.", "답변 생성 완료 했습니다."}:
                    print(f"[RAG Listener] 완료 신호 수신 (skip checklist): {answer}")
                else:
                    stt_result_q.put(answer)
                    ws.send(f"checklist,{answer}")
                    print(f"[RAG->Queue] {answer}")
                    time.sleep(0.5)
        except Exception as e:
            print("[RAG Listener] 오류: ", e)
            break

def wav_mode(path):
    """WAV 파일에서 STT 실행"""
    if not os.path.isfile(path):
        print(f"[ERR] 파일을 찾을 수 없습니다: {path}")
        return
    audio_f32 = load_wav_to_float32(path)
    model = build_model(MODEL_SIZE)
    text, lang, ms = transcribe_audio(model, audio_f32, LANG)
    if text:
        nouns = extract_nouns(text)
        print(f"[STT/{lang}] {text} ({ms:.0f}ms) | Nouns: {nouns}")
    else:
        print("[STT] (무음/인식없음)")

def main():
    global ws
    mode = input(">> 입력 모드 선택 (mic / wav / q): ").strip().lower()

    if mode in {"mic", "wav", "tts", "chatlog"}:
        ws = websocket.WebSocket()
        ws.connect("ws://localhost:8000/ws/stt")
        print("WebSocket 연결 완료 → ws://localhost:8000/ws/stt")

    if mode == "mic":
        HOST, PORT_STT = "127.0.0.1", 5005
        rag_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        rag_sock.connect((HOST, PORT_STT))
        print(f"[MAIN] RAG 서버 연결 완료 -> {HOST}:{PORT_STT}")

        threading.Thread(target=stt_worker, args=(rag_sock,), daemon=True).start()
        threading.Thread(target=rag_listener, args=(rag_sock, stt_result_q), daemon=True).start()
        threading.Thread(target=tts_worker, args=(stt_result_q,), daemon=True).start()

        try:
            while True:
                pass
        except KeyboardInterrupt:
            print("\n[프로그램 종료 요청 감지]")
            try:
                print(f"큐에 남아있는 갯수: {stt_result_q.qsize()}")
                while not stt_result_q.empty():
                    item = stt_result_q.get_nowait()
                    print("[Queue] ", item)
            except queue.Empty:
                print("큐가 비어있습니다.")
            sys.exit(0)
    elif mode == "wav":
        path = input(">> WAV 파일 경로: ").strip()
        wav_mode(path)
    elif mode == "tts":
        threading.Thread(target=tts_worker, args=(stt_result_q,), daemon=True).start()
        msg = input().strip()
        stt_result_q.put(msg)
        # 프로그램이 바로 끝나지 않도록 루프 유지
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("종료")
    elif mode == "chatlog":
        print("💬 CHATTING LOG 입력 모드 (형식: user, 메시지 or bot, 메시지 or checklist, 메시지)")
        print("종료하려면 exit 입력")

        while True:
            line = input(">> ").strip()
            if line.lower() == "exit":
                break

            if "," not in line:
                print("[CHATLOG] 'user, 메시지' 또는 'bot, 메시지' 또는 'checklist, 메시지' 형식으로 입력하세요.")
                continue

            role, msg = line.split(",", 1)
            role = role.strip().lower()
            msg = msg.strip()

            if role not in {"user", "bot", "checklist"}:
                print("[CHATLOG] role은 'user' 또는 'bot' 또는 'checklist'여야 합니다.")
                continue

            ws.send(f"{role},{msg}")
            print(f"[CHATLOG] {role.upper()} 메시지 전송됨: {msg}")
    else:
        print("종료")

if __name__ == "__main__":
    main()