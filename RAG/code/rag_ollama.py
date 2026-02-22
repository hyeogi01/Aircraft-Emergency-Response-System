# rag_ollama_repl.py
# ------------------------------------------------------------
# - REPL 모드: query를 반복 입력받아 RAG(+BM25+E5) → Ollama llama3:8b 호출
# - 문서/컨텍스트 디버그 출력 없음. 최종 답변만 출력
# - 종료 단어 입력 시 루프 종료 후 'ollama stop <model>'로 GPU 메모리 해제
# ------------------------------------------------------------

import json
import re
import os
import sys
import textwrap
import argparse
import subprocess
from pathlib import Path
from math import isfinite
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import faiss
import requests
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

import socket
import time
import re

# --------------------
# 경로 설정 (필요 시 수정)
# --------------------
OUT_DIR = "../data_new/new_vetor_DB"
E5_DIR = str(Path(OUT_DIR) / "faiss_e5_large_v2")
INDEX_JSONL = "../data_new/data/json_rag_files2/index.jsonl"

# --------------------
# 전역 자료구조
# --------------------
id2content = {}
id2meta = {}
docs_tokens = []      # BM25용 토큰화 문서
doc_ids = []          # BM25용 doc id
bm25 = None

# --------------------
# 번역기 (M2M100)
# --------------------
TRANS_MODEL_NAME = "facebook/m2m100_418M"
trans_tokenizer = AutoTokenizer.from_pretrained(TRANS_MODEL_NAME)
trans_model = AutoModelForSeq2SeqLM.from_pretrained(TRANS_MODEL_NAME).to(
    "cuda" if torch.cuda.is_available() else "cpu"
)
trans_device = next(trans_model.parameters()).device

# 음성 인식에서 나오는 한국어 발음을 영어 용어로 바꿔주는 사전
KOR_TO_ENG_TERMS = {
    "에이피유": "APU",
    "블리드": "Bleed",
    "에어": "Air",
    "엔진": "Engine",
    # 필요하면 계속 추가
}



def normalize_terms(text: str) -> str:
    for kor, eng in KOR_TO_ENG_TERMS.items():
        text = text.replace(kor, eng)
    return text

def translate_ko_to_en(text: str) -> str:
    text = normalize_terms(text)
    trans_tokenizer.src_lang = "ko"
    encoded = trans_tokenizer(text, return_tensors="pt").to(trans_device)
    generated = trans_model.generate(**encoded, forced_bos_token_id=trans_tokenizer.get_lang_id("en"))
    return trans_tokenizer.decode(generated[0], skip_special_tokens=True)

def translate_en_to_ko(text: str) -> str:
    trans_tokenizer.src_lang = "en"
    encoded = trans_tokenizer(text, return_tensors="pt").to(trans_device)
    generated = trans_model.generate(**encoded, forced_bos_token_id=trans_tokenizer.get_lang_id("ko"))
    return trans_tokenizer.decode(generated[0], skip_special_tokens=True)

def is_korean(text: str) -> bool:
    return any('가' <= ch <= '힣' for ch in text)

# --------------------
# 유틸
# --------------------
def tokenize_en(text: str):
    """영숫자 기준 심플 토크나이저 (영어 전용 가정)"""
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def load_faiss_index(out_dir: str):
    index = faiss.read_index(os.path.join(out_dir, "index.faiss"))
    meta = []
    with open(os.path.join(out_dir, "meta.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    with open(os.path.join(out_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    # E5 전용
    model = SentenceTransformer(cfg["model_name"])
    return index, meta, cfg, model


def prepare_query_text_for_e5(query: str) -> str:
    # E5는 'query: ' 프리픽스 권장
    return "query: " + query.strip()


def dense_search_e5(query: str, top_k: int, index, meta, cfg, model):
    qtext = prepare_query_text_for_e5(query)
    q = model.encode(
        [qtext],
        normalize_embeddings=cfg.get("normalize", True),
        show_progress_bar=False
    ).astype("float32")
    scores, idxs = index.search(q, top_k)
    idxs = idxs[0].tolist()
    scores = scores[0].tolist()
    results = []
    for rank, (i, sc) in enumerate(zip(idxs, scores), start=1):
        if i == -1:
            continue
        m = meta[i]
        results.append({
            "id": m["id"],  # meta에 'id' 필드가 있다고 가정
            "rank": rank,
            "score": float(sc),
            "source_file": m.get("source_file", ""),
            "title": m.get("title", ""),
            "_retriever": "e5"
        })
    return results


def bm25_search(query: str, top_k: int = 20):
    q_tokens = tokenize_en(query)
    scores = bm25.get_scores(q_tokens)
    idx_sorted = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = []
    for rank, i in enumerate(idx_sorted, start=1):
        cid = doc_ids[i]
        results.append({
            "id": cid,
            "rank": rank,
            "score": float(scores[i]) if isfinite(scores[i]) else 0.0,
            "source_file": id2meta[cid]["source_file"],
            "title": id2meta[cid]["title"],
            "_retriever": "bm25"
        })
    return results


def rrf_fuse(*ranked_lists, k: int = 60, top_k: int = 10):
    """
    Reciprocal Rank Fusion
    ranked_lists: 각 원소는 [{'id','rank',...}, ...] 형태의 리스트
    """
    agg = defaultdict(lambda: {"rrf": 0.0, "candidates": []})
    for lst in ranked_lists:
        for item in lst:
            rid = item["id"]
            rnk = item["rank"]
            rrf = 1.0 / (k + rnk)
            agg[rid]["rrf"] += rrf
            agg[rid]["candidates"].append(item)

    fused = []
    for cid, bundle in agg.items():
        any_item = max(bundle["candidates"], key=lambda x: x.get("score", 0))
        fused.append({
            "id": cid,
            "fused_rrf": bundle["rrf"],
            "source_file": any_item["source_file"],
            "title": any_item["title"],
        })

    fused.sort(key=lambda x: x["fused_rrf"], reverse=True)
    return fused[:top_k]


def hybrid_retrieve(query: str,
                    top_k_bm25: int = 30,
                    top_k_dense: int = 30,
                    rrf_k: int = 60,
                    final_k: int = 10,
                    e5_pack=None):
    """
    BM25 + E5 결과를 RRF로 결합
    e5_pack = (e5_index, e5_meta, e5_cfg, e5_model)
    """
    e5_index, e5_meta, e5_cfg, e5_model = e5_pack

    bm25_hits = bm25_search(query, top_k=top_k_bm25)
    e5_hits = dense_search_e5(query, top_k=top_k_dense,
                              index=e5_index, meta=e5_meta, cfg=e5_cfg, model=e5_model)
    fused = rrf_fuse(bm25_hits, e5_hits, k=rrf_k, top_k=final_k)
    return fused


def build_context(fused, max_chars=3500):
    """LLM 투입 직전 컨텍스트 문자열 조립 (출력은 하지 않음)"""
    ctxs, used = [], 0
    for r in fused:
        cid = r["id"]
        title = r["title"]
        src = r["source_file"]
        body = id2content.get(cid, "")
        block = f"[TITLE] {title}\n[FILE] {src}\n[CONTENT]\n{body}\n"
        if used + len(block) > max_chars:
            break
        ctxs.append(block)
        used += len(block)
    return "\n\n---\n\n".join(ctxs)


def clamp_context(ctx: str, max_chars: int = 7000) -> str:
    return ctx[:max_chars]


def build_prompt(query: str, context: str) -> list:
    system = """
    You are an AI assistant for Airbus A330 flight crew abnormal and emergency procedures.
    Your ONLY source of truth is the provided Flight Crew Operating Manual (FCOM) and Airplane Flight Manual (AFM) context.

    CRITICAL SAFETY RULES:
    - NEVER invent, guess, or simplify procedures. Use ONLY actions explicitly in the given context.
    - NEVER mix procedures. If the context includes multiple related procedures (e.g., ENG FIRE and ENG SHUT DOWN),
      only output the steps directly related to the user query. Do not merge.
    - ALWAYS include the full checklist from the first to the last relevant action in the context. Do not stop early.
    - Step headers must ALWAYS be imperative action instructions (e.g., "Land as soon as possible", "Shut down the affected engine").
      Never use procedure titles (e.g., "ENG FIRE (IN FLIGHT)") or metadata lines (e.g., STATUS, NOTES) as step headers.
    - ALWAYS include all safety limitations and conditions if present in the manuals (e.g., "MAX FL220", "Avoid icing conditions", "Press RCL").
    - NEVER output decorative separators like "................................................".
    - If the context is incomplete, end with the sentence: "The provided manuals do not contain further steps."

    Output Requirements:

    1. Display Version:
       - Repeat the query at the top.
       - Present the answer as a numbered checklist:
         Step 1 – <top-level imperative action>
             → <sub-action if needed>
             → <sub-action if needed>
         Step 2 – <next top-level action>
             → <sub-action if needed>
       - Number ONLY the top-level steps ("Step 1 –", "Step 2 –").
       - Sub-actions must use indentation and arrows (→), never step numbers.
       - Use technical notations (IDLE, DISCH, pb-sw, ON/OFF, FL220) exactly as written in the manuals.
       - Be concise but precise: each step must be a clear pilot action, not a heading.

    2. Audio Version:
       - Convert the Display Version into short, natural spoken sentences for TTS.
       - Do NOT use step numbers. Each instruction = one sentence.
       - Ensure Audio Version matches the Display Version EXACTLY in meaning (no omissions, no additions).
       - Expand abbreviations only if needed for clarity (e.g., FL220 → "flight level two two zero").
       - Keep it direct, natural, and clear for speech.

    IMPORTANT:
    - Display Version and Audio Version must ALWAYS match in meaning and content.
    - Ignore irrelevant context chunks. If an action is not in the manuals, do not output it.
    """

    user = textwrap.dedent(f"""
    Question:
    {query}

    Context (multiple chunks from AFM/FCOM):
    ---
    {context}
    ---

    Instructions:
    Based ONLY on the provided context, produce the answer in this exact structure:

    Display Version
    Query: <repeat the query>
    Answer:
    Step 1 – <imperative action from the manuals>
        → <sub-action if needed>
        → <sub-action if needed>
    Step 2 – <imperative action from the manuals>
        → <sub-action if needed>
    ...
    (Top-level steps must be imperative action instructions, not document titles.
    Sub-actions must use indentation with arrows only. 
    Do not include decorative separators, notes, or status text.)

    Audio Version
    Answer:
    <Short spoken sentences, one per line.
    No step numbers.
    EXACTLY the same actions as Display Version, in natural speech form.>
    """)

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def ask_ollama_llama3(query: str, context: str,
                      model: str = "mistral:7b-instruct-q4_K_M",
                      host: str = "http://localhost:11434",
                      temperature: float = 0.2,
                      num_ctx: int = 8192,
                      timeout_sec: int = 120) -> str:
    context = clamp_context(context, max_chars=7000)
    messages = build_prompt(query, context)
    resp = requests.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx
            },
            "stream": False
        },
        timeout=timeout_sec
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content", "").strip()


def stop_ollama_model(model: str = "mistral:7b-instruct-q4_K_M"):
    """
    대화 종료 후 GPU 메모리 해제: 'ollama stop <model>'
    """
    try:
        subprocess.run(
            ["ollama", "stop", model],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except FileNotFoundError:
        pass


# =====================
# 초기 로드 (인덱스/문서/BM25)
# =====================
def load_corpus_and_bm25():
    with open(INDEX_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            cid = row["id"]
            title = row.get("title", "") or ""
            body = row.get("content", "") or ""
            src = row.get("source_file", "")
            id2content[cid] = body
            id2meta[cid] = {"source_file": src, "title": title}
            tokens = tokenize_en(f"{title}\n{body}")
            docs_tokens.append(tokens)
            doc_ids.append(cid)
    global bm25
    bm25 = BM25Okapi(docs_tokens)


def repl_loop(e5_pack,
              model_name: str = "mistral:7b-instruct-q4_K_M",
              host: str = "http://localhost:11434",
              topk_bm25: int = 30,
              topk_dense: int = 30,
              rrf_k: int = 60,
              final_k: int = 10,
              quit_token: str = ":q"):
    """
    REPL: 사용자가 종료 단어(quit_token 또는 공통 토큰) 입력 시 종료
    """
    common_quit = {quit_token.lower(), "exit", "quit", "/exit", "/quit", "q"}
    print(f"[RAG-Ollama] 입력 대기 중... 종료하려면 '{quit_token}' (또는 exit/quit) 입력")
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] 입력 중단 감지. 종료합니다.")
            break
        if not query:
            continue
        if query.lower() in common_quit:
            print("[INFO] 종료 명령 감지. 정리 중...")
            break

        # --- 번역 처리 ---
        orig_lang = "ko" if is_korean(query) else "en"
        if orig_lang == "ko":
            query = translate_ko_to_en(query)
        else:
            query = query

        # 검색 → 컨텍스트 → LLM 호출
        try:
            fused = hybrid_retrieve(
                query,
                top_k_bm25=topk_bm25,
                top_k_dense=topk_dense,
                rrf_k=rrf_k,
                final_k=final_k,
                e5_pack=e5_pack
            )
            context = build_context(fused, max_chars=9999999)
            answer = ask_ollama_llama3(
                query=query,
                context=context,
                model=model_name,
                host=host
            )
            print(answer)
        except Exception as e:
            print(f"[ERROR] {e}", file=sys.stderr)

    # 루프 종료 시 Ollama 모델 정리
    stop_ollama_model(model_name)
    print(f"[INFO] '{model_name}' 언로드 요청 완료.")

# junkyu
def send_audio_version(audio_text: str, conn, delay: float = 0.5):

    # 문장 단위로 split ('.' 기준 + 줄바꿈 고려)
    sentences = re.split(r'(?<=[.!?])\s+', audio_text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    print("\n=== Audio Sentences to TTS ===\n")
    for s in sentences:
        print(s)  # 서버에서 확인용 출력
        conn.sendall((s + "\n").encode("utf-8"))
        time.sleep(delay)  # 중간에 텀 주기

    # 마지막에 완료 신호 전송
    conn.sendall(("Answer generated successfully.\n").encode("utf-8"))

# junkyu
def socket_loop(e5_pack,
                model_name:str="mistral:7b-instruct-q4_K_M",
                host="http://localhost:11434",
                port=5005,
                topk_bm25=30,
                topk_dense=30,
                rrf_k=60,
                final_k=10):
    """
    STT 클라이언트 → RAG 서버 (text 수신) → LLM 호출 → 답변 전송
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(1)
    print(f"[RAG] 소켓 서버 시작. 포트 {port} 에서 대기 중...")

    conn, addr = srv.accept()
    print(f"[RAG] 연결됨: {addr}")

    try:
        while True:
            data = conn.recv(1024)
            if not data:
                break
            query = data.decode("utf-8").strip()
            print(f"[RAG] 받은 텍스트: {query}")
            if query.lower() in {"exit", "quit", ":q"}:
                print("[RAG] 종료 명령 수신. 정리 중...")
                break

            orig_lang = "ko" if is_korean(query) else "en"
            if orig_lang == "ko":
                query = translate_ko_to_en(query)
                print(f"[RAG] 번역된 쿼리(EN): {query}")

            try:
                fused = hybrid_retrieve(
                    query,
                    top_k_bm25=topk_bm25,
                    top_k_dense=topk_dense,
                    rrf_k=rrf_k,
                    final_k=final_k,
                    e5_pack=e5_pack
                )
                context = build_context(fused, max_chars=9999999)
                answer = ask_ollama_llama3(
                    query=query,
                    context=context,
                    model=model_name,
                    host=host
                )
                print(f"[RAG] 답변: {answer}")
                
                audio_text = ""
                if "Audio Version:" in answer:
                    audio_text = answer.split("Audio Version:")[-1].strip()
                else:
                    audio_text = answer.strip()

                if audio_text.startswith("Answer:"):
                    audio_text = audio_text[len("Answer:"):].strip()
            
                send_audio_version(audio_text, conn, delay=0.7)

            except Exception as e:
                err_msg = f"[ERROR] {e}"
                print(err_msg)
                # conn.sendall((err_msg + "\n").encode("utf-8"))
    finally:
        conn.close()
        stop_ollama_model(model_name)
        print("[RAG] 연결 종료")

# =====================
# main
# =====================
def main():
    parser = argparse.ArgumentParser(description="RAG + Ollama REPL (BM25 + E5, answer only)")
    parser.add_argument("--model", type=str, default="mistral:7b-instruct-q4_K_M", help="Ollama 모델명")
    parser.add_argument("--ollama_host", type=str, default="http://localhost:11434", help="Ollama 호스트")
    parser.add_argument("--topk_bm25", type=int, default=30)
    parser.add_argument("--topk_dense", type=int, default=30)
    parser.add_argument("--final_k", type=int, default=10)
    parser.add_argument("--rrf_k", type=int, default=60)
    parser.add_argument("--quit_token", type=str, default=":q", help="종료 단어 (기본 ':q')")
    args = parser.parse_args()

    # 코퍼스 & BM25 로드(1회)
    load_corpus_and_bm25()
    # E5 FAISS 인덱스 로드(1회)
    e5_index, e5_meta, e5_cfg, e5_model = load_faiss_index(E5_DIR)

    try:
        # repl_loop(
        #     e5_pack=(e5_index, e5_meta, e5_cfg, e5_model),
        #     model_name=args.model,
        #     host=args.ollama_host,
        #     topk_bm25=args.topk_bm25,
        #     topk_dense=args.topk_dense,
        #     rrf_k=args.rrf_k,
        #     final_k=args.final_k,
        #     quit_token=args.quit_token
        # )
        # junkyu
        socket_loop(
            e5_pack=(e5_index, e5_meta, e5_cfg, e5_model),
            model_name=args.model,
            host=args.ollama_host,
            port=5005
        )
    finally:
        # 혹시 중간 예외로 빠져도 정리
        stop_ollama_model(args.model)

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print("[INFO] PyTorch GPU 메모리 해제 완료.")


if __name__ == "__main__":
    main()