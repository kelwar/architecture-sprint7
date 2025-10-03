import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from faiss import IndexFlatL2


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
def get_relevant_chunk(query, index, model, chunks, top_k=1):
    embs = model.encode([query])
    D, I = index.search(x=embs, k=top_k)
    return chunks[I[0][0]]


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
paths = get_files_in_directory_os("./knowledge_base")
chunks = []
for file_path in paths:
    file_chunks = preprocess_data(file_path)
    for chunk in file_chunks:
        chunks.append(chunk)

embeddings = get_embeddings(model, [chunk.page_content for chunk in chunks])
index = IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

print("Relevant chunk:\n", get_relevant_chunk("""Как зовут старшего сына Патрика Грейтхаунда""", index, model, chunks, 10))
print("Relevant chunk:\n", get_relevant_chunk("""Главный город в Шуйтере""", index, model, chunks, 10))