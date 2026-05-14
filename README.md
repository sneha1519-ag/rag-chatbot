# RAG Text-to-Insight Chatbot

> Ask questions about your own documents in plain English — no SQL, no technical knowledge needed.

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot that lets anyone query custom PDF and text datasets using natural language. Built to solve a real business problem: making data locked in documents instantly accessible to non-technical stakeholders.

---

## Why I built this

Most business insights are buried in PDFs, reports, and text files that require manual searching or technical SQL skills to access. This tool eliminates that gap — a team member can ask *"What were the top issues in Q3?"* and get a precise, sourced answer in seconds.

---

**Example queries:**
```
User: What are the key findings from the report?
Bot:  Based on your documents, the key findings are: [1] ...

User: Summarise the risks mentioned in section 3.
Bot:  Section 3 identifies three primary risks: ...
```

---

## How it works

```
Your PDFs / .txt files
        │
        ▼
  [ingest.py] — loads, chunks, and embeds documents
        │
        ▼
  FAISS vector store (local, fast retrieval)
        │
   User query
        │
        ▼
  Semantic search → top-k relevant chunks
        │
        ▼
  OpenAI LLM → grounded, context-aware answer
        │
        ▼
     Response
```

---

## Tech stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12 |
| LLM orchestration | LangChain |
| Embeddings | OpenAI `text-embedding-ada-002` |
| Vector store | FAISS (local, no infra needed) |
| Document parsing | PyPDF2, plain text |
| Interface | CLI (extensible to FastAPI/Streamlit) |

---

## Getting started

### Prerequisites
- Python 3.10+
- OpenAI API key

### Installation

```bash
git clone https://github.com/sneha1519-ag/rag-chatbot.git
cd rag-chatbot
pip install -r requirements.txt
```

### Configuration

```bash
# Create a .env file
echo "OPENAI_API_KEY=your_key_here" > .env
```

### Usage

```bash
# Step 1: Ingest your documents
python ingest.py --source ./your_documents/

# Step 2: Start the chatbot
python app.py
```

Place your `.pdf` or `.txt` files in the `your_documents/` folder. The chatbot will only answer from those sources.

---

## Project structure

```
rag-chatbot/
├── app.py          # Main chatbot interface
├── ingest.py       # Document loading, chunking, and embedding
├── utils.py        # Helper functions
├── requirements.txt
└── .gitignore
```

---

## What I learned

- How vector similarity search (cosine distance in FAISS) retrieves semantically relevant chunks rather than keyword matches
- Chunking strategy matters: overlapping chunks preserve context across boundaries
- Grounding LLM responses in retrieved documents dramatically reduces hallucination vs. open-ended prompting

---

## Roadmap

- [ ] Streamlit UI for non-technical users
- [ ] Multi-document source comparison ("what does document A say vs document B?")
- [ ] Metadata filters (filter by date, source, author)
- [ ] FastAPI endpoint for integration into larger pipelines

---

## Related skills demonstrated

`RAG Architecture` · `Prompt Engineering` · `Vector Databases` · `LLM Integration` · `Python` · `Data Pipeline Design`

---

*Built by [Sneha Agarwal](https://github.com/sneha1519-ag) · Open to feedback and contributions*

