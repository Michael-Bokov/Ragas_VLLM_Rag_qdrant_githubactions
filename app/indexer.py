import os
from pathlib import Path
from typing import List

import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

# Конфигурация из переменных окружения
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "my_docs")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# Инициализация клиента Qdrant
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Инициализация модели эмбеддингов
encoder = SentenceTransformer(EMBEDDING_MODEL)

# def read_documents(folder: str) -> List[str]:
#     """Читает все текстовые файлы из папки (поддерживает .txt, .pdf, .docx)."""
#     texts = []
#     for ext in ("*.txt", "*.pdf", "*.docx","*.md"):
#         for filepath in Path(folder).glob(ext):
#             if ext == "*.md":
#                 with open(filepath, "r", encoding="utf-8") as f:
#                     texts.append(f.read())
#             # Добавьте обработку PDF/DOCX при необходимости
#     return texts
def read_documents(folder: str) -> List[str]:
    texts = []
    path = Path(folder)
    # Список расширений, которые мы считаем текстовыми
    text_extensions = {".txt", ".md"}
    
    print(f"🔍 Начинаю поиск файлов в {folder}...")
    
    # Итерируемся по всем файлам в папке
    for filepath in path.iterdir():
        if filepath.suffix.lower() in text_extensions:
            print(f"📄 Читаю файл: {filepath.name}")
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content.strip():
                        texts.append(content)
                    else:
                        print(f"⚠️ Файл {filepath.name} пуст, пропускаю.")
            except Exception as e:
                print(f"❌ Ошибка при чтении {filepath.name}: {e}")
        else:
            print(f"⏭️ Пропускаю файл (неподдерживаемый формат): {filepath.name}")
            
    print(f"📚 Итого загружено документов: {len(texts)}")
    return texts

def chunk_text(text: str, chunk_size=500, overlap=50) -> List[str]:
    """Разбивает текст на чанки (простая реализация)."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def main():
    # Удаляем коллекцию, если она существует (для переиндексации)
    # try:
    #     client.delete_collection(COLLECTION_NAME)
    # except:
    #     pass
    collections = client.get_collections().collections
    exists = any(c.name == COLLECTION_NAME for c in collections)
    if not exists:
        vector_size = encoder.get_sentence_embedding_dimension()
        print(f"Размерность эмбеддингов: {vector_size}")
        # Создаём коллекцию
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

        # Читаем и обрабатываем документы
        docs_folder = "/app/docs"
        documents = read_documents(docs_folder)

        points = []
        for doc_idx, doc_text in enumerate(documents):
            chunks = chunk_text(doc_text)
            embeddings = encoder.encode(chunks).tolist()
            for chunk_idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"doc{doc_idx}_chunk{chunk_idx}"))
                points.append(
                    models.PointStruct(
                        id=point_id, #f"doc{doc_idx}_chunk{chunk_idx}",
                        vector=emb,
                        payload={"text": chunk, "source": f"doc{doc_idx}"},
                    )
                )
            
        # Загружаем пачками по 100
        for i in range(0, len(points), 100):
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points[i:i+100],
            )

        print(f"Индексация завершена. Загружено {len(points)} чанков.")

if __name__ == "__main__":
    main()