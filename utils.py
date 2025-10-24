# src/utils.py
import os
from typing import List
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def load_text_files(folder: str = "data") -> List[Document]:
    docs = []
    if not os.path.exists(folder):
        print(f"[utils] folder '{folder}' not found")
        return docs
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        if fname.lower().endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
            docs.extend(loader.load())
        elif fname.lower().endswith(".pdf"):
            # PyPDFLoader reads PDF into LangChain Documents
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        # add other file types if needed
    print(f"[utils] Loaded {len(docs)} documents from {folder}")
    return docs

def chunk_documents(docs: List[Document], chunk_size: int = 800, chunk_overlap: int = 100) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    print(f"[utils] Split into {len(chunks)} chunks (chunk_size={chunk_size})")
    return chunks

def build_faiss_store(chunks: List[Document], openai_api_key: str, persist_path: str = "faiss_store"):
    print("[utils] Building embeddings and FAISS vectorstore — this may take a while...")
    embed_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
    store = FAISS.from_documents(chunks, embed_model)
    os.makedirs(persist_path, exist_ok=True)
    store.save_local(persist_path)
    print(f"[utils] Saved FAISS vectorstore to '{persist_path}' with {len(chunks)} chunks")
    return store

def load_faiss_store(openai_api_key: str, persist_path: str = "faiss_store"):
    embed_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
    if not os.path.exists(persist_path):
        raise FileNotFoundError(f"Vectorstore not found at '{persist_path}'. Run ingest first.")
    store = FAISS.load_local(persist_path, embed_model)
    print(f"[utils] Loaded FAISS vectorstore from '{persist_path}'")
    return store
