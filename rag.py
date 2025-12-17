from dotenv import load_dotenv
load_dotenv()

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

FAISS_PATH = "data/faiss"

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def ingest_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    if os.path.exists(FAISS_PATH):
        db = FAISS.load_local(
            FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        db.add_documents(chunks)
    else:
        db = FAISS.from_documents(chunks, embeddings)

    db.save_local(FAISS_PATH)


def answer_question(question: str, chat_history: list):
    if not os.path.exists(FAISS_PATH):
        return "No documents uploaded yet."

    db = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = db.similarity_search(question, k=3)

    context = "\n\n".join(d.page_content for d in docs)
    history = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history])

    prompt = f"""
You are a helpful assistant.
Answer ONLY from the context.

Chat history:
{history}

Context:
{context}

Question: {question}
Answer:
"""

    response = llm.invoke(prompt)
    return response.content
