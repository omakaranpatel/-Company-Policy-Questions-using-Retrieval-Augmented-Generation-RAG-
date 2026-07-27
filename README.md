# Company-Policy-Questions-using-Retrieval-Augmented-Generation-RAG-

# 📋 AI-Powered Employee Policy & Reimbursement RAG Assistant

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework-Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![LLM-Groq](https://img.shields.io/badge/LLM-Groq%20%2F%20OpenAI-green.svg)](https://groq.com/)
[![VectorDB-Chroma](https://img.shields.io/badge/VectorDB-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![License-MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade **Retrieval-Augmented Generation (RAG)** system designed to answer employee policy, expense reimbursement, and HR queries with **100% strict grounding** and **auditable citations**.

Powered by **Groq API** (`llama-3.3-70b-versatile`) and **OpenAI**, featuring a **Hybrid Vector + BM25 Retrieval Engine**, **Cross-Encoder Reranking**, and an intuitive **Streamlit Web Application**.

---

## 🌟 Key Features

| Capability | Technical Implementation |
| :--- | :--- |
| **High-Speed LLM Answers** | Powered by **Groq API** (`llama-3.3-70b-versatile`) or **OpenAI** (`gpt-4o-mini`) with streaming response support. |
| **Hybrid Search Engine** | Combines **Dense Vector Search** (`ChromaDB` + `all-MiniLM-L6-v2`) and **Sparse Lexical Search** (`BM25`) using Reciprocal Rank Fusion (RRF). |
| **Cross-Encoder Reranking** | Re-scores candidate chunks with `ms-marco-MiniLM-L-6-v2` for precise context ordering. |
| **Strict Anti-Hallucination** | System prompts enforce **Zero Outside Knowledge**. Queries without policy context output explicit fallback messaging. |
| **Auditable Citations** | Every answer includes clickable in-text sources and expandable document excerpt drawers. |
| **Header-Aware Ingestion** | Section-aware Markdown and PDF chunking preserves policy tables, caps, and approval matrices intact. |
| **Multi-Turn Conversation** | Retains recent conversation turns for context-aware follow-up question resolution. |
| **Graceful Degradation** | Runs in offline extractive fallback mode if no API key is provided. |

---

## 🏗️ Architecture & Pipeline Flow

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│ Policy Documents│────▶│ Header Ingestion │────▶│ Chroma Vector Store │
│ (PDF / MD / TXT)│     │  & Sub-Chunker   │     │ (all-MiniLM-L6-v2)  │
└─────────────────┘     └──────────────────┘     └──────────┬──────────┘
                                 │                          │
                                 ▼                          │
                        ┌──────────────────┐                │
                        │    BM25 Index    │◀───────────────┘
                        └────────┬─────────┘
                                 │
User Query ──▶ Query Expansion ──┴──▶ Hybrid Retriever ──▶ Cross-Encoder Reranker
                                                                   │
                                                                   ▼
                                                       ┌───────────────────────┐
                                                       │   Answer Generator    │
                                                       │ (Groq / OpenAI / Fall)│
                                                       └───────────┬───────────┘
                                                                   ▼
                                                        Streamlit Answer UI
```

### Pipeline Workflow Steps
1. **Document Loading**: Section-aware splitting on headers (`##`), sub-chunking long policy sections into 512-character blocks with 64-character overlap.
2. **Dual Indexing**: Embeds chunks using `all-MiniLM-L6-v2` stored in ChromaDB alongside a local BM25 index.
3. **Hybrid Retrieval**: Queries expanded with domain synonyms (`per diem`, `nightly cap`), retrieving top candidate hits from both vector and BM25 search fused with weighted RRF scores.
4. **Cross-Encoder Reranking**: Pairs candidate chunks with the user prompt to produce precise re-ordered relevance scores.
5. **Grounded Generation**: Passes top 5–6 reranked chunks to Groq / OpenAI LLMs under a strict anti-hallucination prompt system.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- **Python 3.10** or higher
- **Groq API Key** (Recommended, free tier available at [console.groq.com](https://console.groq.com/)) or **OpenAI API Key**

### 2. Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/<your-username>/policy-rag-assistant.git
cd policy-rag-assistant

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration (`.env`)

Create a `.env` file in the project root:

```env
# Groq API Configuration (Recommended)
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# OpenAI API Configuration (Optional alternative)
OPENAI_API_KEY=

# Retrieval & Model Settings
EMBEDDING_MODEL=all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Document Chunking Parameters
CHUNK_SIZE=512
CHUNK_OVERLAP=64

# Search Weights
TOP_K_VECTOR=12
TOP_K_BM25=12
TOP_K_FINAL=6
HYBRID_VECTOR_WEIGHT=0.6
HYBRID_BM25_WEIGHT=0.4
```

### 4. Index Policy Documents

Run the CLI document indexer to build your local Chroma vector store and BM25 index:

```bash
python scripts/index_documents.py
```

*Options*:
- `--incremental`: Index only newly added policy files.
- `--ocr`: Enable OCR processing for scanned PDF policy documents.
- `--documents-dir <path>`: Specify custom document directory.

### 5. Launch the Streamlit Web Application

```bash
streamlit run app.py
```

Open **`http://localhost:8501`** in your browser.

---

## 🧪 Testing & Verification

Run the automated unit test suite to verify ingestion, hybrid retrieval, and pipeline components:

```bash
python -m unittest discover tests
```

---

## 📂 Policy Documents Included

Sample policy documents located in the `documents/` folder:

| File Name | Description | Key Policy Areas Covered |
| :--- | :--- | :--- |
| `travel_policy.md` | Corporate Travel Policy | Domestic/international hotel caps, flight class approvals, per diem allowances |
| `expense_policy.md` | Reimbursement Policy | Expense report deadlines (30-day rule), non-reimbursable items |
| `finance_policy.md` | Finance & Procurement Policy | Corporate card rules, approval matrices by spend threshold |
| `employee_handbook.md` | Employee Handbook | Annual wellness stipends ($600/year), remote work allowances |

---

## 💡 Sample Queries & Expected Outputs

### Q1: What is the hotel cap for domestic travel?
- **Answer**: The domestic lodging cap is **$200 per night**. Extended stays over 14 nights require corporate housing when available.
- **Sources**: `Travel Policy — 3. Booking Guidelines` (`HR-TRV-2024-v2`)

### Q2: How long do I have to submit an expense report?
- **Answer**: Expense reports must be submitted within **30 days** of incurring the expense. Submissions older than **90 days** require CFO approval.
- **Sources**: `Expense Reimbursement Policy — Submission Requirements` (`FIN-EXP-2024-v3`)

### Q3: Can I expense my gym membership?
- **Answer**: Gym memberships are **not reimbursable** via standard expense reports. However, full-time employees receive a **$600 annual wellness stipend** via the Benefits Portal.
- **Sources**: `Expense Policy — Non-Reimbursable`; `Employee Handbook — Wellness Stipend`

### Q4: What is the pet travel reimbursement policy?
- **Answer**: *I could not find sufficient information in the provided policy documents to answer this question.*

---

## 📁 Repository Structure

```
.
├── app.py                      # Streamlit web interface & chat UI
├── documents/                  # Policy source files (PDF, Markdown, TXT)
│   ├── employee_handbook.md
│   ├── expense_policy.md
│   ├── finance_policy.md
│   └── travel_policy.md
├── scripts/
│   └── index_documents.py      # CLI script for document ingestion & indexing
├── src/
│   ├── config.py               # Centralized configuration & environment loader
│   ├── models.py               # Pydantic data schemas (DocumentChunk, Citation, etc.)
│   ├── pipeline.py             # End-to-end RAG orchestrator
│   ├── utils.py                # Stream patching and environment utilities
│   ├── ingestion/
│   │   ├── loader.py           # Document parsers (PDF, MD, TXT, OCR)
│   │   └── chunker.py          # Header-aware Markdown & token chunker
│   ├── retrieval/
│   │   ├── vector_store.py     # ChromaDB vector store wrapper
│   │   ├── bm25_index.py       # BM25 lexical search implementation
│   │   ├── hybrid_retriever.py # RRF score fusion retriever
│   │   ├── query_expansion.py  # Domain synonym expander
│   │   └── reranker.py         # Cross-encoder candidate reranker
│   └── generation/
│       └── answer_generator.py # Groq / OpenAI LLM integration & fallback mode
├── tests/                      # Unit test suite
├── .gitignore                  # Git ignore settings (excludes .env & data/)
├── requirements.txt            # Python dependencies
└── README.md                   # Repository documentation
```

---

## 🛠️ Tech Stack

- **UI Framework**: Streamlit
- **LLM Engine**: Groq API (`llama-3.3-70b-versatile`), OpenAI (`gpt-4o-mini`)
- **Vector Database**: ChromaDB
- **Embeddings**: HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Lexical Search**: `rank-bm25`
- **PDF Extraction**: `pypdf` / `pdf2image` / `pytesseract`

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
