from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_chain import EnterpriseRAGBot

# Initialize FastAPI application
app = FastAPI(
    title="DBA Internal Enterprise RAG API",
    description="Grounded QA microservice powered by ChromaDB, FlashRank Reranker, and local Ollama",
    version="1.0.0"
)

# Load the RAG pipeline once at startup
rag_bot = EnterpriseRAGBot()

class ChatRequest(BaseModel):
    question: str

@app.post("/api/v1/chat")
def chat_endpoint(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    return rag_bot.ask(req.question)

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "llama3.2"}
