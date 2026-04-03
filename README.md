# ⬡ DocRAG — LLM-Powered Document Q&A System

> **Built a Retrieval-Augmented Generation (RAG) system enabling contextual Q&A over documents using embeddings and LLMs.**

A production-ready RAG pipeline that lets you upload PDFs, DOCX, or text files and ask natural language questions — getting accurate, source-cited answers powered by OpenAI embeddings + GPT.

---

## 🚀 Features

| Feature | Description |
|---|---|
| 📄 Multi-format upload | PDF, DOCX, TXT support |
| 🔍 Semantic search | FAISS vector similarity over embedded chunks |
| 🧠 LLM answers | GPT-4o-mini with RAG context injection |
| 💬 Chat memory | Last 6 turns of conversation passed to the model |
| 📎 Source highlighting | Every answer cites the exact chunks retrieved |
| 📚 Multi-doc support | Query across multiple documents simultaneously |
| ⚡ Fast ingestion | Chunked embedding with overlap for better retrieval |

---

## 🗂 Project Structure

```
genai-rag-qa/
├── app.py              # Streamlit UI — chat interface + sidebar
├── ingest.py           # Document parsing, chunking, embedding, FAISS indexing
├── vector_store/
│   ├── index.faiss     # FAISS index (auto-generated)
│   └── metadata.json   # Chunk metadata (auto-generated)
├── utils/
│   ├── retriever.py    # Cosine similarity search over FAISS
│   ├── llm.py          # OpenAI chat completion with context + history
│   └── helpers.py      # Source formatting, text utilities
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🛠 Tech Stack

- **Python 3.11+**
- **Streamlit** — UI
- **OpenAI API** — `text-embedding-3-small` for embeddings, `gpt-4o-mini` for answers
- **FAISS** (CPU) — local vector store, no external DB needed
- **pypdf + python-docx** — document parsing

---

## ⚙️ Setup

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/genai-rag-qa.git
cd genai-rag-qa
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate      # Mac/Linux
# venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your OpenAI API key

```bash
cp .env.example .env
# Edit .env and paste your key:
# OPENAI_API_KEY=sk-...
```

### 5. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🔄 How It Works

```
Upload PDF/DOCX
      │
      ▼
  Extract text  (pypdf / python-docx)
      │
      ▼
  Chunk text    (512 chars, 64 overlap)
      │
      ▼
  Embed chunks  (OpenAI text-embedding-3-small)
      │
      ▼
  Store in FAISS (cosine similarity index)
      │
      ▼
  User asks question
      │
      ▼
  Embed query → FAISS search → top-k chunks
      │
      ▼
  GPT-4o-mini (question + context + history)
      │
      ▼
  Answer + source citations shown in UI
```

---

## 📸 Screenshot

> _Upload a document → ask questions → get sourced answers_

---

## 🧩 Extending This Project

- **Swap vector DB**: Replace FAISS with [Pinecone](https://pinecone.io) or [Qdrant](https://qdrant.tech) for cloud-hosted search
- **Swap LLM**: Change `model` in `utils/llm.py` to `gpt-4o`, or use [LiteLLM](https://litellm.ai) for open-source models
- **Add OCR**: Use `pytesseract` for scanned PDFs
- **Deploy**: Push to [Streamlit Cloud](https://streamlit.io/cloud) — set `OPENAI_API_KEY` in Secrets

---

## 📝 Resume Line

> *Built a Retrieval-Augmented Generation (RAG) system enabling contextual Q&A over documents using OpenAI embeddings, FAISS vector search, and GPT — with source citation and multi-turn chat memory.*

---

## 📄 License

MIT
