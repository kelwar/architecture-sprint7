import os
import time

import faiss
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


def get_files_in_directory_os(directory_path='.'):
    result = []
    for entry in os.listdir(directory_path):
        if entry.endswith(".txt"):
            full_path = os.path.join(directory_path, entry)
            if os.path.isfile(full_path):
                result.append(full_path)
    return result


def preprocess_data(file_path):
    return (RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
            .split_documents([page for page in TextLoader(file_path).load()]))


def get_embeddings(model, chunks):
    return model.encode(chunks)


# Функция для поиска релевантного фрагмента к запросу пользователя
def get_relevant_chunks(query, index, chunks, top_k=1):
    faiss.normalize_L2(query)
    distances, ids = index.search(query, k=top_k)
    return [chunks[ids[0][i]] for i in range(top_k)]


start_time = time.time()
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
paths = get_files_in_directory_os("../knowledge_base")
chunks = []
for file_path in paths:
    file_chunks = preprocess_data(file_path)
    for chunk in file_chunks:
        chunks.append(chunk)

print(f"Количество чанков для обработки: {len(chunks)}")
embeddings = get_embeddings(model, [chunk.page_content for chunk in chunks])
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
print(f"Время создания индекса: {time.time() - start_time:.2f} с.")

faiss.write_index(index, "faiss.index")

for query in ["""Принцесса Песчаной страны""", """Старший брат Кроуси"""]:
    print("Запрос: ", query)
    start_time = time.time()
    relevant_chunks = get_relevant_chunks(model.encode([query]), index, chunks, 10)
    print(f"Время поиска релевантного чанка: {time.time() - start_time:.2f} с.")
    print("Relevant chunks:")
    for i, chunk in enumerate(relevant_chunks):
        print(i + 1, chunk)
    print()