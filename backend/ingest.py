import os
import fitz  # PyMuPDF
import shutil
from PIL import Image
import io
import random
import time
from dotenv import load_dotenv
import numpy as np
from google.cloud import vision
from google.api_core.exceptions import GoogleAPICallError, RetryError
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from tqdm import tqdm


# ✅ 환경변수 로드 (.env 파일 필요)
load_dotenv()

#------------------phoenix api------------------------------------------------------------------------
# Phoenix 설정
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = "api_key=f3046c156853d98babc:ca2a364"
os.environ["PHOENIX_CLIENT_HEADERS"] = "api_key=f3046c156853d98babc:ca2a364"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"

from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from phoenix.evals import OpenAIModel, QAEvaluator, run_evals
import pandas as pd

# Phoenix 등록 및 계측기 설정
tracer_provider = register(project_name="ocr-ingest-pipeline", auto_instrument=True)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

eval_model = OpenAIModel(model="gpt-4o", temperature=0)
qa_evaluator = QAEvaluator(model=eval_model)

# ✅ 평가 함수 정의
# def evaluate_response(query, response_text, reference=None):
#     df = pd.DataFrame({
#         "input": [query],
#         "output": [response_text],
#         "reference": [reference or ""]
#     })

#     results = run_evals(df, evaluators=[qa_evaluator], provide_explanation=True)
#     result = results[0]  # 하나만 평가되므로 0번째

#     return {
#     "score": result["score"].item() if hasattr(result["score"], 'item') else result["score"],
#     "explanation": result["explanation"].item() if hasattr(result["explanation"], 'item') else result["explanation"],
#     "input_excerpt": query.replace("\n", " "),
#     "output_excerpt": response_text.replace("\n", " ")
# }

#--------------------------------------------------------------------------------------------------------



# ✅ 클라이언트 초기화
vision_client = vision.ImageAnnotatorClient()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ Google Vision OCR 함수 (이미지 저장 없이 OCR만)
def ocr_page_google_vision(page, max_retries=20, base_delay = 3):
    # 1. 고해상도 변환 (4배 확대)
    zoom_x = 1.0
    zoom_y = 1.0
    mat = fitz.Matrix(zoom_x, zoom_y)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # 2. PIL 이미지 → 바이너리 변환
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # 3. Vision API 요청
    image = vision.Image(content=img_byte_arr)
    retries = 0
    while retries < max_retries:
        try:
            response = vision_client.document_text_detection(image=image, timeout=30)

            if response.error.message:
                raise Exception(f"Google Vision API 오류: {response.error.message}")

            return response.full_text_annotation.text

        except (GoogleAPICallError, RetryError, TimeoutError, Exception) as e:
            print(f"🚨 Vision API 호출 실패 (시도 {retries + 1}/{max_retries}): {e}")
            delay = base_delay * (2 ** retries) + random.uniform(0, 1)
            print(f"⏳ {round(delay, 2)}초 후 재시도합니다...")
            time.sleep(delay)
            retries += 1

    print("❌ 최대 재시도 횟수를 초과했습니다. 이 페이지는 건너뜁니다.")
    return ""

# ✅ OpenAI LLM으로 본문 정리 함수
def fix_text_with_llm(ocr_text):
    prompt = f"""
다음은 OCR을 통해 추출된 텍스트입니다.

이 텍스트는 문장 순서가 자연스럽지 않을 수 있습니다.  
하지만 **텍스트에 포함된 모든 내용, 단어, 문장**은 **절대 추가, 삭제, 수정하지 말고 그대로 유지**해야 합니다.
본래 표 형태였던 데이터가 있을 수 있습니다. 표 형태의 정보는 다시 표 형태로 복원해주세요

**오직 문장의 순서만 자연스럽게 재배치**하세요.  
글자, 단어, 문장을 수정하거나 누락시키면 안 됩니다.  
조금이라도 내용이 바뀌어서는 안 됩니다.

[OCR 추출 텍스트]
{ocr_text}

[요구사항]
- 단어, 문장을 추가, 삭제, 수정하지 말 것
- 순서만 자연스럽게 재배열할 것
- 원본 텍스트의 정보를 100% 그대로 유지할 것
- 의미를 바꾸지 말 것
- 띄어쓰기나 간단한 연결 정도는 허용

[결과]
"""
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "너는 문서 정리 전문가야. 본문만 남기고 나머지는 삭제해야 해."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    fixed_text = response.choices[0].message.content.strip()

    print("\n🔢 토큰 사용량 (Prompt):", response.usage.prompt_tokens)
    print("🔢 토큰 사용량 (Completion):", response.usage.completion_tokens)
    print("🔢 토큰 사용량 (Total):", response.usage.total_tokens)

    return fixed_text

# ✅ PDF 읽고 텍스트 분할
def load_and_split_pdfs(pdf_folder):
    texts = []
    evaluation_results = []

    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            doc = fitz.open(os.path.join(pdf_folder, filename))
            for page_num, page in enumerate(doc):
                if page.get_images():  # ✅ 페이지에 이미지가 있으면
                    ocr_text = ocr_page_google_vision(page)
                    fixed_text = fix_text_with_llm(ocr_text)
                    # eval_result = evaluate_response(ocr_text, fixed_text)
                    # evaluation_results.append(eval_result)
                    page_text = fixed_text
                else:
                    # 텍스트만 있는 페이지는 그대로 사용
                    page_text = page.get_text()
                    # 📌 텍스트 페이지도 평가 추가
                    fixed_text = fix_text_with_llm(page_text)
                    # eval_result = evaluate_response(page_text, fixed_text)
                    # evaluation_results.append(eval_result)
                    page_text = fixed_text

                texts.append(page_text)

                # # ✅ 테스트용 출력 (앞부분 500자만 보기)
                # print(f"\n[📄 {filename} - Page {page_num+1}]")
                # print(page_text)  # 길면 500자까지만
                # print("-" * 80)
            doc.close()

    # ✅ 텍스트 조각 분할
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.create_documents(texts), evaluation_results

# ✅ 텍스트를 임베딩하고 FAISS에 저장
def embed_and_save(docs, db_path="vectorstore"):
    embeddings = OpenAIEmbeddings()
    if os.path.exists(os.path.join(db_path, "index.faiss")):
        db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        db.add_documents(docs)
    else:
        db = FAISS.from_documents(docs, embeddings)

    db.save_local(db_path)

# ✅ 메인 실행
if __name__ == "__main__":
    pdf_folder = "data"  # PDF 파일 넣어둔 폴더
    after_folder = "data_after"
    docs, evaluation_results = load_and_split_pdfs(pdf_folder)
    embed_and_save(docs)
    print(" 벡터스토어 생성 완료! (vectorstore 폴더 저장)")

    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            src = os.path.join(pdf_folder, filename)
            dst = os.path.join(after_folder,filename)
            try:
                shutil.move(src,dst)
            except Exception as e:
                print(f"{filename} 이동 실패: {e}")
    
    # ✅ 전체 평가 결과 DataFrame으로 변환 후 출력
    # df_result = pd.DataFrame(evaluation_results)
    # print("\n📋 전체 평가 요약 (score만 표로):")
    # print(df_result[["score"]].to_markdown(index=True))
    # print("\n📋 상세 결과 (각 row별 전체 설명):")
    # for i, row in df_result.iterrows():
    #     print(f"\n📝 [응답 {i+1}]")
    #     print(f"📊 정확도 점수: {row['score']}")
    #     print(f"📥 질문 내용:\n{row['input_excerpt']}")
    #     print(f"📤 모델 응답:\n{row['output_excerpt']}")
    #     print(f"🧠 평가 설명:\n{row['explanation']}")
    #     print("-" * 80)