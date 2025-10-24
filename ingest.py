# ingest.py
import os
from dotenv import load_dotenv
from utils import load_text_files, chunk_documents, build_faiss_store

if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("❌ OPENAI_API_KEY not found in .env file")

    print("🚀 Starting ingestion...")
    docs = load_text_files("data")
    chunks = chunk_documents(docs)
    build_faiss_store(chunks, openai_api_key=api_key)
    print("✅ Ingestion complete! Vector store saved successfully.")
