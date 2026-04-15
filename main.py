from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

DB_PATH = "vector_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:3b"

TOP_K = 8
MAX_DOCS = 5
MAX_CONTEXT_CHARS = 5000
RELEVANCE_THRESHOLD = 1.2

app = FastAPI(title="Career RAG Assistant")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

vector_db = FAISS.load_local(
    DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0
)


def deduplicate_docs(docs):
    seen = set()
    unique_docs = []

    for doc in docs:
        key = (
            doc.metadata.get("job_title", "Unknown Title"),
            doc.metadata.get("company_name", "Unknown Company"),
            doc.metadata.get("source", "Unknown")
        )

        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    return unique_docs


def filter_docs_by_query(results, query):
    q = query.lower()

    web_keywords = [
        "web", "frontend", "front-end",
        "backend", "back-end",
        "full stack", "full-stack"
    ]

    if any(word in q for word in web_keywords):
        strong_filtered = []

        for doc, score in results:
            title = doc.metadata.get("job_title", "").lower()

            if (
                "web developer" in title
                or "frontend" in title
                or "front-end" in title
                or "backend" in title
                or "back-end" in title
                or "full stack" in title
                or "full-stack" in title
            ):
                strong_filtered.append((doc, score))

        if strong_filtered:
            return strong_filtered

    return results


def choose_docs(results):
    if not results:
        return [], "general_only"

    strong = []
    weak = []

    for doc, score in results:
        if score <= RELEVANCE_THRESHOLD:
            strong.append((doc, score))
        else:
            weak.append((doc, score))

    if strong:
        selected = [doc for doc, _ in strong[:MAX_DOCS]]
        return deduplicate_docs(selected), "documents"

    selected = [doc for doc, _ in weak[:3]]
    return deduplicate_docs(selected), "weak_documents"


def build_context(docs):
    parts = []
    total = 0

    for i, doc in enumerate(docs, start=1):
        chunk = f"[Document {i}]\n{doc.page_content.strip()}\n"

        if total + len(chunk) > MAX_CONTEXT_CHARS:
            break

        parts.append(chunk)
        total += len(chunk)

    return "\n\n".join(parts)


def build_prompt(query, context, mode):
    if mode in ("documents", "weak_documents"):
        return f"""
You are a helpful career assistant.

The context contains job descriptions.

Your task:
- Extract the answer from the documents only.
- Focus ONLY on the most important and commonly mentioned qualifications or skills.
- Ignore rare, niche, or less important items.

Rules:
- Prioritize:
  - education
  - experience
  - essential programming skills
  - core web development tools or requirements
- Do NOT list every single tool or technology.
- Group similar items together instead of listing many examples.
- Avoid over-detail.
- Only include information that is clearly supported by the retrieved documents.
- Do NOT infer, assume, or imply skills that are not explicitly mentioned.
- Do NOT use phrases like "implied", "likely", "probably", or "may include".
- If something is not mentioned in the documents, do not include it.
- Prefer common qualifications found across the retrieved documents.

Context:
{context}

Question:
{query}
"""
    else:
        return f"""
You are a helpful career assistant.

No reliable document context was found.
Answer using general knowledge.
Keep the answer concise and accurate.

Question:
{query}
"""


def ask_rag(query: str):
    results = vector_db.similarity_search_with_score(query, k=TOP_K)
    results = filter_docs_by_query(results, query)

    selected_docs, mode = choose_docs(results)
    context = build_context(selected_docs)
    prompt = build_prompt(query, context, mode)

    response = llm.invoke(prompt)

    sources = []
    for doc in selected_docs:
        sources.append({
            "job_title": doc.metadata.get("job_title", "Unknown Title"),
            "company_name": doc.metadata.get("company_name", "Unknown Company"),
            "source": doc.metadata.get("source", "Unknown")
        })

    return {
        "answer": response.content,
        "sources": sources,
        "mode": mode
    }


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"answer": None, "sources": [], "query": ""}
    )


@app.post("/", response_class=HTMLResponse)
async def ask_page(request: Request, query: str = Form(...)):
    query = query.strip()

    if not query:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "answer": "Please enter a valid question.",
                "sources": [],
                "query": ""
            }
        )

    result = ask_rag(query)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "answer": result["answer"],
            "sources": result["sources"],
            "query": query
        }
    )


@app.post("/api/ask")
async def ask_api(payload: dict):
    query = str(payload.get("query", "")).strip()

    if not query:
        return JSONResponse(
            status_code=400,
            content={"error": "Query is required."}
        )

    return ask_rag(query)