# 🚀 Career RAG Assistant

A Retrieval-Augmented Generation (RAG) powered AI assistant that analyzes job descriptions to extract skills, qualifications, and insights using FastAPI, FAISS, and LLMs.

---

## 🧠 Overview

This project builds an intelligent career assistant that:

- Retrieves relevant job descriptions from a vector database
- Uses an LLM to extract structured insights
- Focuses on skills, qualifications, and tools
- Reduces hallucination by grounding answers in real data

---

## ⚙️ Tech Stack

- FastAPI
- FAISS
- LangChain
- HuggingFace Embeddings
- Ollama (LLM)
- Jinja2 + HTML/CSS

---

## 📂 Project Structure

```
rag_project/
│
├── load_data.py
├── ingest.py
├── main.py
│
├── data/
├── vector_db/
│
├── templates/
│   └── index.html
│
└── static/
    └── style.css
```

---

## 🚀 How to Run

### 1. Install dependencies

```
py -m pip install -r requirements.txt
```

### 2. Load dataset

```
py load_data.py
```

### 3. Build vector database

```
py ingest.py
```

### 4. Run FastAPI server

```
py -m fastapi dev main.py
```

### 5. Open in browser

```
http://127.0.0.1:8000
```

API Docs:

```
http://127.0.0.1:8000/docs
```

---

## 💡 Example Questions

```
what are the most required skills for web development
```

```
what qualifications are required for a web developer
```

```
what tools are commonly used in frontend jobs
```

---

## 🔍 Features

- Retrieval-Augmented Generation (RAG)
- Multi-document analysis
- Smart filtering for web-related roles
- Reduced hallucination
- Source tracking (job title, company)

---

## 🧠 How It Works

1. Load and clean job descriptions  
2. Split into chunks  
3. Convert to embeddings  
4. Store in FAISS  
5. Retrieve relevant documents  
6. Send context to LLM  
7. Generate grounded answer  

---

## ⚠️ Notes

- Uses local LLM via Ollama  
- Embedding model: sentence-transformers/all-MiniLM-L6-v2  
- Some warnings can be ignored  

---

## 📈 Future Improvements

- Chat-style interface  
- Skill ranking system  
- Better filtering  
- React frontend  

---

## 👨‍💻 Author

Built as a first RAG project to explore LLMs and AI systems.

---

## ⭐

If you like it, give it a star!
