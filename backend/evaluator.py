# evaluator.py
import os
import pandas as pd
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.evals import (
    OpenAIModel,
    QAEvaluator,
    RelevanceEvaluator,
    HallucinationEvaluator,
    ToxicityEvaluator,
    run_evals
)
from dotenv import load_dotenv

load_dotenv()

# ✅ 환경 설정
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = "api_key=f3046c156853d98babc:ca2a364"
os.environ["PHOENIX_CLIENT_HEADERS"] = "api_key=f3046c156853d98babc:ca2a364"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"

# ✅ 트레이싱 설정 (전역에서 단 1번만 호출)
tracer_provider = register(project_name="my-rag-chatbot", auto_instrument=True)
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

# ✅ 평가 모델 초기화
eval_model = OpenAIModel(model="gpt-4o", temperature=0)

#평가자 리스트
evaluators = {
    QAEvaluator(model=eval_model),
    RelevanceEvaluator(model=eval_model),
    HallucinationEvaluator(model=eval_model),
    ToxicityEvaluator(model=eval_model)
}


import pprint
# ✅ 평가 함수 정의
def evaluate_response(query: str, response_text: str, reference: str = "") -> dict:
    df = pd.DataFrame({
        "input": [query],
        "output": [response_text],
        "reference": [reference]
    })

    results = run_evals(df, evaluators=evaluators, provide_explanation=True)
    results = pd.concat(results, ignore_index=True)
    return results