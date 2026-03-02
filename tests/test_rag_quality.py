import pytest
import requests
import json
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas.metrics._faithfulness import faithfulness
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_recall import context_recall
from langchain_huggingface import HuggingFaceEmbeddings as RagasHFEmbeddings
from datasets import Dataset
from langchain_openai import ChatOpenAI # Ragas дружит с этим форматом
from ragas.run_config import RunConfig

from openai import OpenAI
from ragas.llms import llm_factory

from ragas.llms import LangchainLLMWrapper

from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer

# Настройки подключения к твоим запущенным контейнерам
RAG_URL = "http://localhost:8080/ask"
VLLM_URL = "http://localhost:8000/v1"

@pytest.fixture
def MY_embeddings():
    # Модель должна быть доступна локально или скачана
    hf_e = RagasHFEmbeddings(model_name="BAAI/bge-m3", model_kwargs={'device': 'cuda'})#SentenceTransformer("BAAI/bge-m3")#
    return LangchainEmbeddingsWrapper(hf_e)    
@pytest.fixture
def MY_llm():
    # Обертка vLLM для Ragas (он будет оценивать сам себя)
    # client = OpenAI(
    # #     base_url=VLLM_URL,
    # #     api_key="EMPTY",
    # # )
    #     api_key="AQVNy849xPo2rksyO6VesSCidTA2dSJyIhTtAT2K",
    #     base_url="https://llm.api.cloud.yandex.net/v1",
    #     default_headers={
    #         "Authorization": "ajebpous9uhmlp5j4et5",
    #         "OpenAI-Project": "gpt://ajec270584uftr6tccgq/yandexgpt-lite"}
    #     )

    # client = OpenAI(base_url=VLLM_URL, api_key="EMPTY")
    # return llm_factory(model="ModelCloud/Meta-Llama-3.1-8B-Instruct-gptq-4bit", client=client,temperature=0.0, 
    #     max_tokens=1024,extra_headers={"stop": ["\t", "\x0c"]})
    
    # client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    # return llm_factory(
    #     model="llama3.2:3b-instruct-q4_K_M", 
    #     client=client,
    #     temperature=0.0
    llm = ChatOpenAI(
        model="ModelCloud/Meta-Llama-3.1-8B-Instruct-gptq-4bit",
        openai_api_base=VLLM_URL,
        openai_api_key="EMPTY",
        temperature=0,
        #max_tokens=1050, # Give it enough space for JSON
        # #model_kwargs={
        #     # THIS IS KEY: Forces vLLM to strictly follow JSON grammar
        #     "extra_body": {"guided_json": "json_object"} 
        # }
    )
    return LangchainLLMWrapper(llm)
    # )
    
    # return ChatOpenAI(
    #     model="llama3.1:8b-instruct-q4_K_M", 
    #     openai_api_base="http://localhost:11434/v1",
    #     openai_api_key="ollama", 
    #     temperature=0,
    #     max_tokens=2048,
    #     timeout=180,
    #     model_kwargs={"response_format": {"type": "json_object"}}
    # )
    # return ChatOllama(
    #     model="llama3.2:3b-instruct-q4_K_M", 
    #         temperature=0,
    #         format="json",
    #         keep_alive=0 
    # )
    #)
def test_rag_quality(MY_llm,MY_embeddings):
    # 1. Загружаем твои 10-20 голденов
    with open("tests/goldens.json", "r") as f:
        goldens = json.load(f)

    questions, answers, contexts, grounds = [], [], [], []

    # 2. Прогоняем через живой RAG
    for g in goldens:
        res = requests.post(RAG_URL, json={"query": g["question"], "top_k": 3}).json()
        
        questions.append(g["question"])
        answers.append(res["answer"])
        contexts.append([c[:3000] for c in res["contexts"]])#Добавь contexts в Response Model FastAPI# Обрезаем каждый чанк до 2000 символов
        grounds.append(g["ground_truth"])

    # 3. Формируем датасет для Ragas
    # data = {
    #     "question": questions,      
    #     "answer": answers,          
    #     "contexts": contexts,       
    #     "ground_truth": grounds     
    # }
    data = {
            "user_input": questions,
            "response": answers,
            "retrieved_contexts": contexts,
            "reference": grounds
        }
    dataset = Dataset.from_dict(data)

    # 4. Оценка (LLM-as-a-Judge)
    result = evaluate(
        dataset,
        metrics=[faithfulness,context_recall,answer_relevancy,], #  answer_relevancy, 
        llm=MY_llm,
        embeddings=MY_embeddings,
    #     column_map={
    #     "question": "user_input",
    #     "answer": "response",
    #     "contexts": "retrieved_contexts",
    #     "ground_truth": "reference"
    # },
        run_config=RunConfig(max_workers=1, timeout=200) # 1 поток и длинный таймаут
    )

    # 5. Quality Gates (Критерии прохождения теста)
    #print(result)
    df = result.to_pandas()  # теперь result - это Dataset, а не list!
    # print(df.columns)
    # print(df["user_input"])
    # print(df["retrieved_contexts"])
    # print(df['response'])
    # print(df['reference'])
    # print(df['answer_relevancy'])
    numeric_cols = df.select_dtypes(include=['number']).columns
    mean_scores = df[numeric_cols].fillna(0).mean().to_dict()
    print("Средние значения метрик:")
    for metric, value in mean_scores.items():
        print(f"  {metric}: {value:.4f}")
    #mean_scores = df.mean().to_dict()
    #print(f"Ragas Scores: {mean_scores}")
    print(result.to_pandas()[["faithfulness",  "context_recall","answer_relevancy"]]) #"answer_relevancy",
    assert mean_scores["faithfulness"] > 0.6, f"Галлюцинации выше порога! Score: {mean_scores['faithfulness']:.3f} (нужно > 0.7)"
    assert mean_scores["context_recall"] > 0.6, f" Полнота контекста недостаточна: {mean_scores['context_recall']:.3f} (нужно > 0.6)"
    #assert mean_scores["answer_relevancy"] > 0.8, f"Релевантность низкая! Score: {mean_scores['answer_relevancy']:.3f} (нужно > 0.8)"

