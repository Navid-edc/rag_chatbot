from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from flashrank import Ranker, RerankRequest

# Configuration constants matching ingest.py
CHROMA_DB_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class AdvancedRetriever:
    def __init__(self):
        # 1. Load the cached local embedding model from Mac SSD
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"local_files_only": True}
        )
        # 2. Connect to the existing ChromaDB created in Layer 2
        self.vector_db = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=self.embeddings
        )
        # 3. Initialize lightweight local Cross-Encoder Reranker (ms-marco-MiniLM-L-12-v2)
        self.reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./reranker_cache")

    def retrieve_and_rerank(self, query: str, initial_k: int = 5, top_n: int = 2):
        # Stage 1: Fast Vector Search (Retrieve top 5 candidate chunks)
        initial_docs = self.vector_db.similarity_search_with_score(query, k=initial_k)
        
        print(f"--- Stage 1: ChromaDB Vector Search (Top {len(initial_docs)} Candidates) ---")
        passages = []
        for idx, (doc, distance) in enumerate(initial_docs):
            print(f"Candidate {idx+1} [L2 Distance: {distance:.4f}]: {doc.page_content[:65]}...")
            passages.append({
                "id": idx + 1,
                "text": doc.page_content,
                "meta": doc.metadata
            })

        # Stage 2: Deep Cross-Encoder Reranking (Score how well each chunk actually answers the query)
        rerank_request = RerankRequest(query=query, passages=passages)
        reranked_results = self.reranker.rerank(rerank_request)

        # Keep only the top_n highest-scoring chunks
        final_chunks = reranked_results[:top_n]

        print(f"\n--- Stage 2: After FlashRank Reranking (Top {top_n} Winners) ---")
        for rank, item in enumerate(final_chunks, 1):
            # Note: In Reranking, HIGHER score (closer to 1.0) means BETTER relevance!
            print(f"Rank {rank} [Relevance Score: {item['score']:.4f} (Higher is better!)]:")
            print(f"   -> {item['text']}\n")

        return final_chunks

if __name__ == "__main__":
    retriever = AdvancedRetriever()
    test_q = "How many days of annual leave do employees get and where do they submit it?"
    print(f"\nTesting Query: '{test_q}'\n")
    retriever.retrieve_and_rerank(test_q)
