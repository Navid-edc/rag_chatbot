from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_chain import EnterpriseRAGBot
from agent import EnterpriseAgent

app = FastAPI(
    title="DBA Internal Enterprise RAG & Agent API",
    description="Hybrid QA microservice featuring Two-Stage RAG (ChromaDB + FlashRank) and Tool-Calling SQL Agent via local Ollama",
    version="2.0.0"
)

# Initialize both pipelines at startup
rag_bot = EnterpriseRAGBot()
agent_bot = EnterpriseAgent()

class ChatRequest(BaseModel):
    question: str

class AgentRequest(BaseModel):
    session_id: str = "default_session"
    question: str

@app.post("/api/v1/chat")
def rag_chat_endpoint(req: ChatRequest):
    """Direct two-stage RAG endpoint for policy lookup."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    return rag_bot.ask(req.question)

@app.post("/api/v1/agent")
def agent_chat_endpoint(req: AgentRequest):
    """Stateful tool-calling Agent endpoint combining RAG policy search and SQL employee records."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    return agent_bot.chat(req.session_id, req.question)

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "llama3.2", "capabilities": ["rag", "sql_agent"]}
