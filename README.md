# Enterprise RAG & QA Microservice

An enterprise-grade, fully local Retrieval-Augmented Generation (RAG) service designed for internal company documentation. Built with **LangChain (LCEL)**, **ChromaDB**, **FlashRank Cross-Encoder**, and local open-weight LLMs via **Ollama (Llama 3.2)**, exposed as a production-ready **FastAPI** REST interface.

## 1. System Architecture
1. **Stage 1 (Bi-Encoder Vector Search):** Fast candidate retrieval (Top-5) from ChromaDB using `sentence-transformers/all-MiniLM-L6-v2` (384-dim).
2. **Stage 2 (Cross-Encoder Reranking):** Deep relevance scoring via FlashRank (`ms-marco-MiniLM-L-12-v2`) to select the Top-2 chunks.
3. **Stage 3 (Grounded Generation):** Deterministic generation (`temperature=0.0`) with strict hallucination guardrails via local Ollama (`llama3.2`).

## 2. Case Study: Fixing Title-Chunk Ranking Bias
During error analysis on the query *"How many days of annual leave do employees get?"*, standard L2 vector search ranked the short document header (`[Document 1: Annual Leave...]`, L2=0.7603) higher than the actual policy paragraph containing the answer (`26 days`, L2=0.8199) due to length bias. 
Adding the **FlashRank Cross-Encoder** resolved this completely, assigning a **0.9993** relevance score to the actual policy paragraph and promoting it to Rank 1.

## 3. Quick Start & Evaluation
- **Ingest Documents:** `python ingest.py`
- **Run Automated Evaluation Suite (100% Accuracy):** `python evaluate.py`
- **Start FastAPI Server:** `uvicorn app:app --host 127.0.0.1 --port 8000`
