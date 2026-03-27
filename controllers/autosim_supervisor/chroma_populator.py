import json
import os
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

SHREDDED_FILE = "shredded_rag_chunks.json"
DB_DIR = "./chroma_db"


def is_valid_chunk(text: str) -> bool:
    """
    Validates a text chunk by ensuring it meets minimum length requirements
    and does not contain restricted keywords.

    Args:
        text (str): The text chunk to validate.

    Returns:
        bool: True if the chunk is valid, False otherwise.
    """
    if not text:
        return False

    # Ensure the chunk has a minimum of 40 words
    if len(text.split()) < 40:
        return False

    bad_patterns = ["figure", "copyright", "table of contents"]
    text_lower = text.lower()

    # Reject chunks containing restricted patterns
    if any(p in text_lower for p in bad_patterns):
        return False

    return True


def populate_db():
    """
    Loads pre-processed text chunks, filters them, constructs Document objects,
    and populates a local Chroma vector database using OpenAI embeddings.
    """
    print("Starting vector database population...")
    print("-" * 50)

    # 1. Load the pre-processed chunks
    if not os.path.exists(SHREDDED_FILE):
        print(f"Error: Required input file '{SHREDDED_FILE}' not found.")
        return

    with open(SHREDDED_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Successfully loaded {len(chunks)} raw chunks.")

    # 2. Convert raw chunks to Document objects and filter out invalid ones
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

        doc = Document(
            page_content=content.strip(),
            metadata=metadata,
        )

        docs.append(doc)
        # Create a unique composite ID for the document to prevent duplication
        ids.append(f"{metadata['source']}_{metadata['chunk_id']}")

    print(f"Filtering complete. {len(docs)} high-quality chunks retained.")

    if not docs:
        print("Error: No valid documents remaining to embed after filtering.")
        return

    # 3. Initialize Embeddings
    print("Connecting to OpenAI embeddings API (model: text-embedding-3-small)...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 4. Initialize and populate the Vector Database with batching
    print(f"Initializing Chroma database at '{DB_DIR}'...")
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    # Batch insert to ensure performance and stability (avoiding API timeouts)
    BATCH_SIZE = 100

    for i in range(0, len(docs), BATCH_SIZE):
        batch_docs = docs[i : i + BATCH_SIZE]
        batch_ids = ids[i : i + BATCH_SIZE]

        print(f"Processing batch: records {i} to {i + len(batch_docs) - 1}...")
        vector_db.add_documents(documents=batch_docs, ids=batch_ids)

    print("-" * 50)
    print("Vector database populated and persisted successfully.")
    print(f"Storage path: {DB_DIR}")
    print(f"Total embeddings stored: {len(docs)}")


if __name__ == "__main__":
    populate_db()
