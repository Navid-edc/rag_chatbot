import os
import shutil
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Define file paths for our knowledge base and local vector database
DOCS_PATH = "company_docs.txt"
CHROMA_DB_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def build_vector_db():
    print("1. Loading internal DBA company documents...")
    # Load the raw text file into LangChain Document objects
    loader = TextLoader(DOCS_PATH, encoding="utf-8")
    documents = loader.load()

    print("2. Splitting documents into smaller semantic chunks...")
    # Use RecursiveCharacterTextSplitter to keep paragraphs and sentences intact
    # chunk_size=350 characters, chunk_overlap=50 characters to preserve context at boundaries
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   -> Created {len(chunks)} chunks from the source document.")

    print(f"3. Initializing local Hugging Face Embedding model ({EMBEDDING_MODEL_NAME})...")
    # Convert text chunks into 384-dimensional dense numerical vectors locally on Mac
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME,model_kwargs={"local_files_only": True})

    # Remove old database directory if it exists to avoid duplicate chunks on re-runs
    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)

    print("4. Creating and persisting local Chroma Vector Database...")
    # Store chunks and their embeddings inside ./chroma_db folder
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    print("   -> Vector database successfully built and saved to './chroma_db'!\n")
    return vector_db

def test_vector_search(vector_db):
    # Test query to verify semantic similarity search works
    sample_query = "How many days of annual leave do employees get and where do they submit it?"
    print(f"5. Testing Vector Search with query:\n   '{sample_query}'\n")
    
    # Retrieve top 2 most similar chunks based on vector cosine distance
    results = vector_db.similarity_search_with_score(sample_query, k=3)
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"--- Result {i} (Distance Score: {score:.4f}) ---")
        print(doc.page_content)
        print()

if __name__ == "__main__":
    db = build_vector_db()
    test_vector_search(db)
