import os
from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI

app = FastAPI()

# Конфигурация
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "my_docs")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
VLLM_HOST = os.getenv("VLLM_HOST", "localhost")
VLLM_PORT = int(os.getenv("VLLM_PORT", 8000))

# Клиенты
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
encoder = SentenceTransformer(EMBEDDING_MODEL)
llm_client = OpenAI(
    base_url=f"http://{VLLM_HOST}:{VLLM_PORT}/v1",
    api_key="EMPTY",  # vLLM не требует ключа
)

class Question(BaseModel):
    query: str
    top_k: int = 5

class Answer(BaseModel):
    answer: str
    sources: list[str]
    contexts: list[str]

@app.post("/ask", response_model=Answer)
def ask(question: Question):
    # 1. Получаем эмбеддинг вопроса
    query_emb = encoder.encode(question.query).tolist()

    # 2. Ищем похожие чанки в Qdrant
    search_result = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_emb,
        limit=question.top_k,
    )

    # 3. Формируем контекст из найденных текстов
    context_parts = []
    contexts = []
    sources_map = {} # Для хранения соответствия ID -> Название файла
    
    for i, hit in enumerate(search_result):
        source_name = hit.payload.get("source", "unknown")
        ref_id = i + 1
        sources_map[ref_id] = source_name
        
        contexts.append(hit.payload["text"])
        # Формируем блок текста с метаданными для модели
        context_parts.append(f"[Источник №{ref_id}, файл: {source_name}]\nТекст: {hit.payload['text']}")

    context_text = "\n\n---\n\n".join(context_parts)
    messages = [
        {
            "role": "system", 
            "content": (
                "Ты — ассистент-астроном. Твоя задача — отвечать на вопросы, используя ТОЛЬКО предоставленный контекст. "
                "ОБЯЗАТЕЛЬНО указывай номер источника в скобках после утверждения, например: (Источник №1). "
                "Если в контексте нет ответа, так и скажи: 'В предоставленных документах нет информации об этом'."
            )
        },
        {
            "role": "user", 
            "content": f"КОНТЕКСТ:\n{context_text}\n\nВОПРОС: {question.query}"
        }
    ]
    # 5. Отправляем запрос в vLLM
    response = llm_client.chat.completions.create(
        model="ModelCloud/Meta-Llama-3.1-8B-Instruct-gptq-4bit", #"TheBloke/Llama-3.1-8B-Instruct-GPTQ",  # должно с моделью vLLM
        messages=messages,
        max_tokens=512,
        temperature=0.2,
    )
    #answer_text = response.choices[0].text.strip()
    answer_text = response.choices[0].message.content.strip()
    final_sources = list(set(sources_map.values()))
    return Answer(answer=answer_text, sources=final_sources, contexts=contexts)

@app.get("/health")
def health():
    return {"status": "ok"}