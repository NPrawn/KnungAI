import os
import sqlite3
import json
import numpy as np
import pandas as pd

from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from evaluator import evaluate_response

# ✅ 환경변수 로드
load_dotenv()

# ✅ FastAPI 앱 생성
app = FastAPI(
    title="학교 공지사항 기반 RAG 챗봇 (FAQ 캐시 + FAISS LangChain RAG)"
)

# ✅ CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙
app.mount("/static", StaticFiles(directory="../frontend/build/static"), name="static")

# ✅ SQLite 연결
conn = sqlite3.connect("faq_cache.db", check_same_thread=False)
cursor = conn.cursor()

# ✅ 테이블 생성 (질문 + 본문 context + 임베딩 + 사용횟수)
cursor.execute("""
    CREATE TABLE IF NOT EXISTS faq_cache (
        question TEXT PRIMARY KEY,
        context TEXT,
        embedding BLOB,
        count INTEGER DEFAULT 1
    )
""")
conn.commit()

# ✅ Pydantic 모델
class Message(BaseModel):
    type: str
    text: str

class QuestionRequest(BaseModel):
    question: str
    history: List[Message] = []

class AnswerResponse(BaseModel):
    answer: str


# - 정보가 없으면 "해당 내용은 공지사항에서 확인되지 않습니다."라고 답변하세요.
# ✅ LangChain 설정
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
당신은 공주대학교 공지사항을 기반으로 행정 정보를 제공하는 챗봇입니다.

- 반드시 아래 본문(context)에서만 정보를 바탕으로 답변하세요.

- 문체는 공식적이고 간결하며, 제목 없는 서술형 단락으로 자연스럽게 작성하세요.
- 가능한 경우 다음과 같은 순서로 내용을 구성하세요:
1. 질문의 핵심에 대한 간단한 설명
2. 관련 조건 및 기준
3. 절차나 신청 방법
4. 필요한 경우 유의사항
- 답변 마지막에는 반드시 다음과 같이 공지사항의 URL을 포함하세요:
출처: 

---

질문: {question}

공지사항 본문 (context):
{context}
"""
)

embeddings = OpenAIEmbeddings()
vectorstore_path = "vectorstore"

# ✅ FAISS 벡터스토어 로드
if os.path.exists(vectorstore_path):
    db = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
else:
    db = FAISS.from_documents([], embeddings)

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# ✅ LangChain RAG QA 체인
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt}
)

# ✅ 임베딩 생성 함수
def get_embedding(text: str):
    return embeddings.embed_query(text)

# ✅ FAQ DB 검색 함수
def find_similar_cached_context(question: str, threshold=0.92):
    question_vector = np.array(get_embedding(question))

    cursor.execute("""
    SELECT question, context, embedding 
    FROM faq_cache
    ORDER BY count DESC
    LIMIT 2""")

    rows = cursor.fetchall()

    best_score = 0
    best_question = None
    best_context = None

    for db_question, db_context, db_embedding_blob in rows:
        if db_embedding_blob is None:
            continue

        db_vector = np.frombuffer(db_embedding_blob, dtype=np.float32)
        cosine_sim = np.dot(question_vector, db_vector) / (np.linalg.norm(question_vector) * np.linalg.norm(db_vector))

        if cosine_sim > best_score:
            best_score = cosine_sim
            best_question = db_question
            best_context = db_context

    if best_score >= threshold and best_context:
        cursor.execute("UPDATE faq_cache SET count = count + 1 WHERE question = ?", (best_question,))
        conn.commit()
        return best_context
    else:
        return None

# ✅ 새 질문 + 본문 context 저장 함수
def save_question_context(question: str, context: str):
    embedding = np.array(get_embedding(question), dtype=np.float32).tobytes()
    cursor.execute(
        "INSERT OR IGNORE INTO faq_cache (question, context, embedding) VALUES (?, ?, ?)",
        (question, context, embedding)
    )
    conn.commit()

#히스토리를 포함한 question 요약
def summarize_question(history: List[dict], current_question: str) -> str:
    history_str = ""
    for msg in history:
        if msg["type"] == "user":
            history_str += f"사용자: {msg['text']}\n"
        elif msg["tpye"] == "bot":
            history_str += f"크눙이: {msg['text']}\n"

        prompt = f"""
너는 “학생 질문 요약 비서”야.
너의 역할은 학생이 마지막으로 한 질문을, 직전 대화 흐름을 고려해서 자연스럽고 구체적인 질문 문장으로 정리해주는 것이야.

⸻

🧠 지켜야 할 규칙
	1.	반드시 마지막 질문(current_question)의 의미만 재구성해.
    → 문장 형태는 바꿔도 되지만, 새로운 정보(예: 자격 요건, 조건 등)는 절대 추가하지 마.
	2.	대화 맥락(history_str)은 참고만 하고, 핵심은 마지막 질문에서 파악해.
	3.	학생 말투로 자연스럽고 간결하게 묻는 질문형 문장으로 정리해.

다음은 지금 까지의 대화야:
{history_str}
마지막 질문:
{current_question}


        """
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        response = llm.invoke(prompt)
        return response.content.strip()

# ✅ 질문 처리 API
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    try:
        last_question = request.question.strip()
        history = [msg.model_dump() for msg in request.history]

        question = summarize_question(history, last_question)

        # 1. FAQ 캐시에서 먼저 비슷한 질문 찾기
        context = False #find_similar_cached_context(question)

        # if context:
        #     # ✅ context 기반으로 LLM 답변 생성
        #     prompt = custom_prompt.format(context=context, question=question)
        #     response = llm.invoke(prompt)
        #     #evaluate---------------------------------------------------------------------------------------------------------------
        #     eval_result = evaluate_response(question, response.content, reference=context)
        #     with open("evaluation_log.txt", "a", encoding="utf-8") as f:
        #         f.wirte(f"\n\n[질문] {question}")
        #         f.write(f"\n[답변] {response.content}")
        #         f.write(f"\n[context] {context}")

        #         for result in eval_result:
        #             f.write(f"\n평가지표: {result['evaluator']}")
        #             f.write(f"\n점수: {result['score']}")
        #             f.write(f"\n설명: {result['explanation']}")
        #         f.write(f"{'-----' * 20}\n")
        #     #-----------------------------------------------------------------------------------------------------------------------

        #     return AnswerResponse(answer=response.content)

        # 2. 못 찾으면 FAISS를 LangChain QA로 검색
        #    (qa_chain이 retriever.get_relevant_documents() + LLM 호출을 같이 해준다)
        relevant_docs = retriever.get_relevant_documents(question)

        if not relevant_docs:
            return AnswerResponse(answer="관련된 공지사항을 찾을 수 없습니다.")

        # ✅ 검색된 문서들 합쳐서 하나의 context 만들기
        context = "\n".join([doc.page_content for doc in relevant_docs])

        # ✅ 만든 context로 LLM에 질문해서 답변 생성
        prompt = custom_prompt.format(context=context, question=question)
        response = llm.invoke(prompt)
        #evaluate---------------------------------------------------------------------------------------------------------------
        eval_result = evaluate_response(question, response.content, reference=context)
        with open("evaluation_log.txt", "a", encoding="utf-8") as f:
            f.write(f"\n\n[질문] {question}")
            f.write(f"\n[답변] {response.content}")
            f.write(f"\n[context] {context}")

            f.write("\n")
            f.write(eval_result[['label', 'score']].to_string(index=False))
            f.write('\n\n')
            for label, explanation in zip(eval_result["label"], eval_result["explanation"]):
                f.write(f"[{label}] \n{explanation}\n")
            f.write(f"\n{'-----' * 20}\n")
        #-----------------------------------------------------------------------------------------------------------------------
        

        # ✅ 질문 + FAISS 검색된 context를 FAQ DB에 저장
        # save_question_context(question, context)

        return AnswerResponse(answer=response.content)

    except Exception as e:
        print(f"❗에러 발생: {e}")
        return Response(
            content=json.dumps({"error": str(e)}, ensure_ascii=False),
            media_type="application/json; charset=utf-8",
            status_code=500
        )

# 루트
# React 정적 빌드 전체를 루트에서 서빙
app.mount("/", StaticFiles(directory="../frontend/build", html=True), name="frontend")