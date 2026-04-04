import json
import os
import pickle
import shutil
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv

load_dotenv()

SHREDDED_FILE = "shredded_rag_chunks.json"
DB_DIR = "./chroma_db"
BM25_FILE = "bm25_retriever.pkl"


def is_valid_chunk(text: str) -> bool:
    """Validates chunk length and removes noise."""
    if not text or len(text.split()) < 40:
        return False

    bad_patterns = ["figure", "copyright", "table of contents"]
    if any(p in text.lower() for p in bad_patterns):
        return False
    return True


def populate_db():
    print("Starting Hybrid Knowledge Base population (Chroma + BM25)...")
    print("-" * 50)

    if not os.path.exists(SHREDDED_FILE):
        print(f"Error: Required input file '{SHREDDED_FILE}' not found.")
        return

    with open(SHREDDED_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Successfully loaded {len(chunks)} raw chunks.")

    docs = []
    ids = []

    for chunk in chunks:
        content = chunk.get("content", "")
        if not is_valid_chunk(content):
            continue

        metadata = {
            "source": chunk.get("source", "unknown"),
            "chunk_id": chunk.get("chunk_id", -1),
            "type": chunk.get("type", "unknown"),
        }

        docs.append(Document(page_content=content.strip(), metadata=metadata))
        ids.append(f"{metadata['source']}_{metadata['chunk_id']}")

    print(f"Filtering complete. {len(docs)} high-quality chunks retained.")

    if not docs:
        print("Error: No valid documents remaining to embed after filtering.")
        return

    # ==========================================
    # 1. Sparse Retrieval: Build & Save BM25 Index
    # ==========================================
    print("Building BM25 Sparse Index (Keyword Matching)...")
    bm25_retriever = BM25Retriever.from_documents(docs)

    # Save the BM25 retriever to disk so the LangChain brain can load it
    with open(BM25_FILE, "wb") as f:
        pickle.dump(bm25_retriever, f)
    print(f"BM25 Index serialized and saved to '{BM25_FILE}'")

    # ==========================================
    # 2. Dense Retrieval: Build ChromaDB
    # ==========================================
    print("Connecting to OpenAI embeddings API (text-embedding-3-small)...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Clear old database to prevent duplication issues during testing
    if os.path.exists(DB_DIR):
        print("Clearing old ChromaDB directory to ensure a clean build...")
        shutil.rmtree(DB_DIR)

    print(f"Initializing new Chroma database at '{DB_DIR}'...")
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    BATCH_SIZE = 100
    for i in range(0, len(docs), BATCH_SIZE):
        batch_docs = docs[i : i + BATCH_SIZE]
        batch_ids = ids[i : i + BATCH_SIZE]
        print(
            f"Processing dense embedding batch: records {i} to {i + len(batch_docs) - 1}..."
        )
        vector_db.add_documents(documents=batch_docs, ids=batch_ids)

    print("-" * 50)
    print("Hybrid Knowledge Base built successfully!")
    print(f"- ChromaDB (Semantic Search) path: {DB_DIR}")
    print(f"- BM25 (Keyword Search) path: {BM25_FILE}")
    print(f"Total documents processed: {len(docs)}")


if __name__ == "__main__":
    populate_db()
