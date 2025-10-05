import os

from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_community.llms.yandex import YandexGPT
from langchain_text_splitters import RecursiveCharacterTextSplitter

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, SecretStr

def get_files_in_directory_os(directory_path='.'):
    result = []
    for entry in os.listdir(directory_path):
        if entry.endswith(".txt"):
            full_path = os.path.join(directory_path, entry)
            if os.path.isfile(full_path):
                result.append(full_path)
    return result

def preprocess_data(splitter, file_path):
    return splitter.split_documents([page for page in TextLoader(file_path).load()])

def get_chunks(splitter):
    paths = get_files_in_directory_os("./knowledge_base")
    result = []
    for file_path in paths:
        file_chunks = preprocess_data(splitter, file_path)
        for chunk in file_chunks:
            result.append(chunk)
    return result

chunks = get_chunks(RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100))
print(f"Количество чанков для обработки: {len(chunks)}")

prompt = PromptTemplate.from_template(
    """f'
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
)
api_key = SecretStr(os.getenv("YC_IAM_TOKEN"))
llm = YandexGPT(model_uri="gpt://b1grfikcp5as92ttdh2d/yandexgpt-5-lite/latest",
                model_name="yandexgpt-lite",
                model_version="latest",
                iam_token=api_key)
llm_chain = prompt | llm

app = FastAPI()

class Query(BaseModel):
    question: str

@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")

@app.post("/ask")
def ask(query: Query):
    return llm_chain.invoke({"context": chunks, "input": query.question})