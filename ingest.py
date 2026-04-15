import re
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
DATA_PATH = "data"
DB_PATH = "vector_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def extract_metadata(text: str) -> dict:
    title_match = re.search(r"Job Title:\s*(.*)", text)
    company_match = re.search(r"Company:\s*(.*)", text)

    return {
        "job_title": title_match.group(1).strip() if title_match else "Unknown Title",
        "company_name": company_match.group(1).strip() if company_match else "Unknown Company",
    }


def main():
    loader = DirectoryLoader(
        DATA_PATH,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents = loader.load()
    print(f"Loaded {len(documents)} documents")

    for doc in documents:
        meta = extract_metadata(doc.page_content)
        doc.metadata["job_title"] = meta["job_title"]
        doc.metadata["company_name"] = meta["company_name"]
        doc.metadata["source"] = doc.metadata.get("source", "Unknown")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120
    )

    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    for chunk in chunks:
        meta = extract_metadata(chunk.page_content)
        chunk.metadata["job_title"] = chunk.metadata.get("job_title", meta["job_title"])
        chunk.metadata["company_name"] = chunk.metadata.get("company_name", meta["company_name"])
        chunk.metadata["source"] = chunk.metadata.get("source", "Unknown")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(DB_PATH)

    print("Vector DB created successfully!")


if __name__ == "__main__":
    main()