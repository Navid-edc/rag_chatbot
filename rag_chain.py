from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from retriever import AdvancedRetriever

# Use local Llama 3.2 served via Ollama
OLLAMA_MODEL = "llama3.2"

class EnterpriseRAGBot:
    def __init__(self):
        # Initialize our two-stage retriever (ChromaDB + FlashRank Cross-Encoder)
        self.retriever = AdvancedRetriever()
        
        # Set temperature=0.0 to enforce deterministic, factual responses and minimize hallucination
        self.llm = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0.0
        )
        
        # Strict system prompt enforcing Grounded Generation and source citation
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an internal AI Assistant for Data Pardazan Bonyan Ava (DBA).\n"
                "Answer the employee's question strictly using ONLY the provided context chunks below.\n"
                "Rules:\n"
                "1. If the answer is not explicitly stated in the context, reply exactly: "
                "'I cannot find this information in the internal DBA documentation.'\n"
                "2. Do not assume, extrapolate, or use outside knowledge.\n"
                "3. Keep your answer concise, professional, and directly reference the policy details.\n\n"
                "Context:\n{context}"
            )),
            ("human", "{question}")
        ])
        
        # Build the LCEL (LangChain Expression Language) generation chain
        self.chain = self.prompt | self.llm | StrOutputParser()

    def ask(self, question: str) -> dict:
        # Retrieve top 5 from vector DB and rerank down to the best 2 chunks
        reranked_chunks = self.retriever.retrieve_and_rerank(question, initial_k=5, top_n=2)
        
        # Format the winning chunks into a single context string for the LLM prompt
        context_text = "\n\n---\n\n".join(
            [f"[Source Chunk {i+1} | Relevance Score: {c['score']:.4f}]:\n{c['text']}" 
             for i, c in enumerate(reranked_chunks)]
        )
        
        print("-> Generating grounded response via local Ollama (llama3.2)...")
        answer = self.chain.invoke({
            "context": context_text,
            "question": question
        })
        
        return {
            "question": question,
            "answer": answer,
            "sources": [
                {"text": c["text"], "relevance_score": float(c["score"])} 
                for c in reranked_chunks
            ]
        }

if __name__ == "__main__":
    bot = EnterpriseRAGBot()
    
    # Test 1: In-domain question (Should be answered accurately from Document 1 & 4)
    q1 = "How many days of annual leave do we get, and how should we submit the request?"
    print(f"\n==================== TEST 1 (In-Domain) ====================")
    res1 = bot.ask(q1)
    print(f"\n[FINAL BOT ANSWER]:\n{res1['answer']}\n")
    
    # Test 2: Out-of-domain question (Testing Hallucination Guardrail)
    q2 = "Does DBA provide free gym memberships or dental insurance?"
    print(f"\n==================== TEST 2 (Guardrail Test) ====================")
    res2 = bot.ask(q2)
    print(f"\n[FINAL BOT ANSWER]:\n{res2['answer']}\n")
