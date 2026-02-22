import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

# Mac/Linux: 설치 후 자동 감지. 경로를 수동 지정하려면 아래 주석 해제 후 확인한 경로 입력
# pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

def pdf_to_markdown(pdf_path, output_md_path):
    doc = fitz.open(pdf_path)
    all_text = []

    for page_num, page in enumerate(doc, start=1):
        # 1) 먼저 텍스트 추출 시도
        text = page.get_text("text")

        if text.strip():  # 텍스트가 있으면 그대로 사용
            all_text.append(text)
        else:
            # 2) OCR fallback
            print(f"[OCR] {pdf_path} - page {page_num}")
            pix = page.get_pixmap(dpi=300)  # 페이지를 이미지로 변환
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img, lang="eng")  # OCR 수행
            all_text.append(text)

    # 결과 저장
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_text))

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_folder, filename)
            md_name = os.path.splitext(filename)[0] + ".md"
            output_md_path = os.path.join(output_folder, md_name)

            print(f"Processing: {pdf_path} -> {output_md_path}")
            pdf_to_markdown(pdf_path, output_md_path)

# =============================
# 실행 예시
# =============================
if __name__ == "__main__":
    input_folder = "/home/piai/바탕화면/posco-academy-30th/ai_project/data_new/data/pdf_rag_files2"   # 변환할 PDF가 들어있는 폴더
    output_folder = "/home/piai/바탕화면/posco-academy-30th/ai_project/data_new/data/md_rag_files2"   # 변환된 MD 파일이 저장될 폴더
    process_folder(input_folder, output_folder)
