from dotenv import load_dotenv
load_dotenv()

import os
from fastapi import FastAPI, UploadFile, File
from rag import ingest_pdf, answer_question
from memory import get_memory, add_message

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Simple Document Q&A")

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = f"{UPLOAD_DIR}/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    ingest_pdf(file_path)
    return {"message": "PDF uploaded and indexed"}

@app.post("/ask")
async def ask(question: str, session_id: str):
    history = get_memory(session_id)
    answer = answer_question(question, history)

    add_message(session_id, "user", question)
    add_message(session_id, "assistant", answer)

    # return {
    #     "answer": answer,
    #     "session_id": session_id
    # }

    return answer
    
    

