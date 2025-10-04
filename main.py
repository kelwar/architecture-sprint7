import os
import time

# from langchain import hub
# from langchain.chains.retrieval_qa.base import RetrievalQA
# from langchain.llms import openai
# from typing import Optional, List, Tuple

from langchain_community.document_loaders import TextLoader
# from langchain_community.llms import openai
from langchain_community.vectorstores import FAISS
# from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from openai import OpenAI
from sentence_transformers import SentenceTransformer
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface.llms import HuggingFaceEndpoint, HuggingFacePipeline
# from langchain_huggingface.chat_models import ChatHuggingFace
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
# from transformers import Pipeline
# from ragatouille import RAGPretrainedModel

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel


def get_files_in_directory_os(directory_path='.'):
    result = []
    for entry in os.listdir(directory_path):
        if entry.endswith(".txt"):
            full_path = os.path.join(directory_path, entry)
            if os.path.isfile(full_path):
                result.append(full_path)
    return result

def preprocess_data(file_path):
    return (RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            .split_documents([page for page in TextLoader(file_path).load()]))

# Функция для поиска релевантного фрагмента к запросу пользователя
def get_relevant_chunks(query, index, chunks, top_k=1):
    # faiss.normalize_L2(query)
    distances, ids = index.search(query, k=top_k)
    return [chunks[ids[0][i]] for i in range(top_k)]

# def answer_with_rag(
#         question: str,
#         llm: Pipeline,
#         knowledge_index: FAISS,
#         reranker: Optional[RAGPretrainedModel] = None,
#         num_retrieved_docs: int = 30,
#         num_docs_final: int = 5,
# ) -> Tuple[str, List[LangchainDocument]]:
#     # Gather documents with retriever
#     print("=> Retrieving documents...")
#     relevant_docs = knowledge_index.similarity_search(query=question, k=num_retrieved_docs)
#     relevant_docs = [doc.page_content for doc in relevant_docs]  # Keep only the text
#
#     # Optionally rerank results
#     if reranker:
#         print("=> Reranking documents...")
#         relevant_docs = reranker.rerank(question, relevant_docs, k=num_docs_final)
#         relevant_docs = [doc["content"] for doc in relevant_docs]
#
#     relevant_docs = relevant_docs[:num_docs_final]
#
#     # Build the final prompt
#     context = "\nExtracted documents:\n"
#     context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)])
#
#     final_prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)
#
#     # Redact an answer
#     print("=> Generating answer...")
#     answer = llm(final_prompt)[0]["generated_text"]
#
#     return answer, relevant_docs


start_time = time.time()
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
paths = get_files_in_directory_os("./knowledge_base")
chunks = []
for file_path in paths:
    file_chunks = preprocess_data(file_path)
    for chunk in file_chunks:
        chunks.append(chunk)

print(f"Количество чанков для обработки: {len(chunks)}")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.from_documents(chunks, embeddings)
retriever = db.as_retriever()
print(f"Время создания индекса: {time.time() - start_time:.2f} с.")

results = db.similarity_search_with_score("Принцесса Песчаной страны", 10)
for i, document in enumerate(results):
    print(i + 1, document)
print()


template = """f'
### Роль
System: Ты помощник, который сначала размышляет, а потом отвечает. Всегда пиши свои шаги. 

### Шаги работы
1. Внимательно прочитай документы.
2. Определи, какие из них действительно релевантны вопросу.
3. Сконспектируй ключевые факты (можешь делать пометки для себя, но не показывай их пользователю).
4. Сформулируй итоговый ответ на русском, опираясь только на подтверждённые факты.

### Формат выдачи
Ответ должен состоять из двух частей:
**A. Краткий ответ** (1‑3 предложения).
**B. Развёрнутое объяснение** (по пунктам), где каждый тезис снабжён ссылкой‑номером на источник в квадратных скобках.

### Примеры
Q: Как называется столица Зибенландов?
A: Столица Зибенландов называется Бухта регента.

Контекст:
<<<
{context}
>>>

Вопрос:
{input}
'"""


prompt = PromptTemplate.from_template(template)
#
# client = OpenAI(base_url="https://openrouter.ai/api/v1",
#                 api_key="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")

llm = ChatOpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-4o-mini"
)
llm_chain = LLMChain(prompt=prompt, llm=llm)

print(llm_chain.run({"context": chunks, "input": "Кто старший брат Кроуси Октопуса?"}))

# llm = HuggingFaceEndpoint(model="gpt2", task="text-generation")
# llm = HuggingFacePipeline.from_model_id(
#     model_id="gpt2",
#     task="text-generation")
# prompt = hub.pull("rlm/rag-prompt")

#
# chain = prompt | completion
# qa_chain = RetrievalQA.from_llm(
#     llm, retriever=db.as_retriever(), prompt=prompt
# )
# qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=db.as_retriever())
# qa_chain = create_stuff_documents_chain(llm, prompt)
# chain = create_retrieval_chain(retriever, qa_chain)

# context = """
# ### Роль
# Ты — крупная русскоязычная LLM‑модель‑ассистент.
# Твоя задача — аккуратно ответить на вопрос пользователя, используя ТОЛЬКО информацию из текстовых документов в директории ./knowledge_base.
# Если в документах нет нужной информации, честно скажи «Не знаю».
# Избегай домыслов и галлюцинаций.
#
# ### Шаги работы
# 1. Внимательно прочитай документы.
# 2. Определи, какие из них действительно релевантны вопросу.
# 3. Сконспектируй ключевые факты (можешь делать пометки для себя, но не показывай их пользователю).
# 4. Сформулируй итоговый ответ на русском, опираясь только на подтверждённые факты.
# 5. В конце ответа проставь цитаты вида [1], [2] — это номера документов из блока <Документы>, которые подтвердили конкретное утверждение.
#
# ### Формат выдачи
# Ответ должен состоять из двух частей:
# **A. Краткий ответ** (1‑3 предложения).
# **B. Развёрнутое объяснение** (по пунктам), где каждый тезис снабжён ссылкой‑номером на источник в квадратных скобках.
# """

# result = chain.invoke({
#     "context": chunks,
#     "input": "Принцесса Песчаной страны"
# })
#
# print(result["answer"])

app = FastAPI()

class Query(BaseModel):
    question: str

@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")

@app.post("/ask")
def ask(query: Query):
    # emb = model.encode([query.question])
    # candidates = get_relevant_chunks(embeddings, index, chunks, 20)
    # candidates = sorted(candidates, key=lambda x: x.cosine_sim, reverse=True)[:15]

    # qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=db.as_retriever())
    # return qa_chain.run(query.question)
    return ""