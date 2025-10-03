import os

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
    loader = TextLoader(file_path)
    pages = [page for page in loader.load()]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=32)
    return text_splitter.split_documents(pages)


def get_embeddings(model, chunks):
    return model.encode(chunks)


# Функция для поиска релевантного фрагмента к запросу пользователя
def get_relevant_chunk(query, index, chunks, top_k=1):
    faiss.normalize_L2(query)
    distances, ids = index.search(query, k=top_k)
    return chunks[ids[0][0]]


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
paths = get_files_in_directory_os("./knowledge_base")
chunks = []
for file_path in paths:
    file_chunks = preprocess_data(file_path)
    for chunk in file_chunks:
        chunks.append(chunk)

embeddings = get_embeddings(model, [chunk.page_content for chunk in chunks])
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "faiss.index")

print("Relevant chunk:\n", get_relevant_chunk(model.encode(["""Принцесса Песчаной страны"""]), index, chunks))
print("Relevant chunk:\n", get_relevant_chunk(model.encode(["""Старший брат Кроуси Октопуса"""]), index, chunks))
print("Relevant chunk:\n", get_relevant_chunk(model.encode(["""Младший брат Кроуси Октопуса"""]), index, chunks))