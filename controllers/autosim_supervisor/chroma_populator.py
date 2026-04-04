import json
import os
import shutil
import pickle
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv

# Load environment variables (OpenAI API Key)
load_dotenv()

# Configuration
INPUT_FILE = "shredded_rag_chunks.json"
CHROMA_DB_DIR = "./chroma_db"
BM25_INDEX_FILE = "./bm25_index.pkl"


def build_hybrid_database():
    """
    Reads the shredded JSON chunks and builds a Hybrid Search infrastructure:
    1. A Dense Vector Database (Chroma) for semantic meaning.
    2. A Sparse Keyword Index (BM25) for exact terminology matching.
    """
    print("=" * 50)
    print("Initiating Hybrid Knowledge Base Construction...")
    print("=" * 50)

    # 1. Load the shredded chunks
    if not os.path.exists(INPUT_FILE):
        print(
            f"[ERROR] Could not find {INPUT_FILE}. Did you run the_shredder.py first?"
        )
        return

    print(f"Loading document chunks from {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        try:
            chunks_data = json.load(f)
        except json.JSONDecodeError:
            print(f"[ERROR] {INPUT_FILE} is corrupted or empty.")
            return

    if not chunks_data:
        print("[WARNING] No chunks found in the input file.")
        return

    # 2. Convert raw JSON dictionaries into LangChain Document objects
    documents = []
    for item in chunks_data:
        # Preserve structural metadata (like headers) from the VLM shredder
        metadata = {
            "source": item.get("source", "unknown"),
            "type": item.get("type", "unknown"),
            "chunk_id": item.get("chunk_id", -1),
        }

        # Add markdown headers to metadata if they exist
        if "headers" in item and isinstance(item["headers"], dict):
            metadata.update(item["headers"])

        doc = Document(page_content=item.get("content", ""), metadata=metadata)
        documents.append(doc)

    print(f"Successfully loaded {len(documents)} LangChain Document objects.")

    # 3. Build the Dense Index (ChromaDB)
    print("\n--- Building Dense Vector Index (ChromaDB) ---")
    if os.path.exists(CHROMA_DB_DIR):
        print("Clearing old ChromaDB directory to prevent duplication...")
        shutil.rmtree(CHROMA_DB_DIR)

    print("Generating embeddings via OpenAI (text-embedding-3-small)...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # This automatically processes the docs, calls the OpenAI API, and saves to disk
    vector_db = Chroma.from_documents(
        documents=documents, embedding=embeddings, persist_directory=CHROMA_DB_DIR
    )
    print(f"Dense vector database saved to '{CHROMA_DB_DIR}'.")

    # 4. Build the Sparse Index (BM25 Keyword Search)
    print("\n--- Building Sparse Keyword Index (BM25) ---")
    print("Calculating TF-IDF keyword frequencies...")

    bm25_retriever = BM25Retriever.from_documents(documents)

    # We must save the BM25 index to disk so the LangChain brain can load it instantly
    with open(BM25_INDEX_FILE, "wb") as f:
        pickle.dump(bm25_retriever, f)

    print(f"Sparse keyword index saved to '{BM25_INDEX_FILE}'.")

    print("\n" + "=" * 50)
    print("Knowledge Base Construction Complete.")
    print("Your AI Brain is now ready for Hybrid Search.")
    print("=" * 50)


if __name__ == "__main__":
    build_hybrid_database()
