# PDF Chat App 📄

A RAG-based conversational app that lets you upload any PDF and ask questions 
about it in plain English. Built with LangChain, ChromaDB, Groq, and Streamlit.

## What it does
- Upload any PDF document
- Ask questions in natural language
- Get accurate answers grounded strictly in the document
- Chat history persists across questions in the same session

## Tech Stack
| Component | Technology |
|---|---|
| LLM | Groq — llama-3.3-70b-versatile |
| Embeddings | sentence-transformers — all-MiniLM-L6-v2 (local) |
| Vector Store | ChromaDB |
| PDF Extraction | pdfplumber |
| Framework | LangChain |
| Frontend | Streamlit |
| Deployment | Streamlit Community Cloud |

## How RAG works here
1. PDF is loaded and split into 1000-character chunks with 200-character overlap
2. Each chunk is embedded locally using all-MiniLM-L6-v2
3. Embeddings stored in ChromaDB
4. User question is embedded and top 3 relevant chunks retrieved
5. Chunks + question sent to Groq LLM for grounded answer

## Setup

### Prerequisites
- Python 3.12+
- Groq API key (free at console.groq.com)

### Installation
```bash
git clone https://github.com/CoderAbhishek/pdf-chat
cd pdf-chat
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Run locally
Create a `.env` file in the project root:

    GROQ_API_KEY=your_key_here

Then run:
```bash
streamlit run app.py
```