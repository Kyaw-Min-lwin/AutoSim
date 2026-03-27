import json
import os
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

SHREDDED_FILE = "shredded_rag_chunks.json"
DB_DIR = "./chroma_db"


# -----------------------------
# 🧠 EXTRA FILTER (FINAL CLEAN PASS)
# -----------------------------
def is_valid_chunk(text):
    if not text:
        return False

    if len(text.split()) < 40:
        return False

    bad_patterns = ["figure", "copyright", "table of contents"]
    text_lower = text.lower()

    if any(p in text_lower for p in bad_patterns):
        return False

    return True


# -----------------------------
# 🧠 BUILD VECTOR DB
# -----------------------------
def populate_db():
    print("==================================================")
    print(" 🧠 THE EMBEDDER V2: BUILDING THE VECTOR VAULT")
    print("==================================================")

    # 1. Load chunks
    if not os.path.exists(SHREDDED_FILE):
        print(f"[ERROR] Missing {SHREDDED_FILE}")
        return

    with open(SHREDDED_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"[LOADER] Loaded {len(chunks)} raw chunks.")

    # 2. Convert to Documents + filter
    docs = []
    ids = []

    for chunk in chunks:
        content = chunk["content"]

        if not is_valid_chunk(content):
            continue

        metadata = {
            "source": chunk.get("source", "unknown"),
            "chunk_id": chunk.get("chunk_id", -1),
            "type": chunk.get("type", "unknown"),
        }

        doc = Document(
            page_content=content.strip(),
            metadata=metadata,
        )

        docs.append(doc)
        ids.append(f"{metadata['source']}_{metadata['chunk_id']}")

    print(f"[CLEAN] {len(docs)} high-quality chunks remain after filtering.")

    if not docs:
        print("[ERROR] No valid documents to embed.")
        return

    # 3. Embeddings
    print("[EMBEDDER] Connecting to OpenAI embeddings...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 4. Create DB with batching
    print(f"[VAULT] Creating Chroma DB at {DB_DIR}...")

    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    # Batch insert (important for performance + stability)
    BATCH_SIZE = 100

    for i in range(0, len(docs), BATCH_SIZE):
        batch_docs = docs[i : i + BATCH_SIZE]
        batch_ids = ids[i : i + BATCH_SIZE]

        print(f"[BATCH] Processing {i} → {i + len(batch_docs)}")

        vector_db.add_documents(documents=batch_docs, ids=batch_ids)

    print("\n[SUCCESS] Vector DB built and persisted!")
    print(f"[PATH] {DB_DIR}")
    print(f"[TOTAL] {len(docs)} embeddings stored.")


if __name__ == "__main__":
    populate_db()
