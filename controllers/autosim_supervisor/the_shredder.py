import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

DOCS_DIR = "./docs"
OUTPUT_FILE = "shredded_rag_chunks.json"


# -----------------------------
# 🧠 SMART CLEANING
# -----------------------------
def clean_text(text):
    lines = text.splitlines()

    cleaned = []
    for line in lines:
        line = line.strip()

        # remove junk / short noise
        if not line:
            continue
        if len(line) < 40:
            continue
        if "copyright" in line.lower():
            continue
        if "figure" in line.lower():
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


# -----------------------------
# 🧠 PARAGRAPH-BASED CHUNKING
# -----------------------------
def smart_chunk(text, max_words=200):
    paragraphs = text.split("\n\n")
    chunks = []

    for p in paragraphs:
        words = p.split()

        if len(words) < 40:
            continue

        # split large paragraphs safely
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i : i + max_words])

            if len(chunk.split()) < 40:
                continue

            chunks.append(chunk.strip())

    return chunks


# -----------------------------
# 🧠 MAIN SHREDDER
# -----------------------------
def shred_documents():
    print("==================================================")
    print(" 🔪 THE SHREDDER V2: PRECISION MODE ACTIVATED")
    print("==================================================")

    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f"[ERROR] Created '{DOCS_DIR}'. Add your files and rerun.")
        return

    all_chunks = []

    for filename in os.listdir(DOCS_DIR):
        filepath = os.path.join(DOCS_DIR, filename)

        # -----------------------------
        # 📄 PDF HANDLING
        # -----------------------------
        if filename.endswith(".pdf"):
            print(f"[PDF] Processing: {filename}")

            loader = PyPDFLoader(filepath)
            pages = loader.load()

            for page in pages:
                cleaned = clean_text(page.page_content)
                chunks = smart_chunk(cleaned)

                for c in chunks:
                    all_chunks.append({"source": filename, "type": "pdf", "content": c})

        # -----------------------------
        # 🧾 JSON HANDLING (ALREADY CLEAN)
        # -----------------------------
        elif filename.endswith(".json"):
            print(f"[JSON] Processing: {filename}")

            with open(filepath, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)

                    for item in data:
                        content = item.get("content", "")
                        source = item.get("source", filename)

                        cleaned = clean_text(content)
                        chunks = smart_chunk(cleaned)

                        for c in chunks:
                            all_chunks.append(
                                {"source": source, "type": "json", "content": c}
                            )

                except Exception as e:
                    print(f"[ERROR] {filename}: {e}")

    if not all_chunks:
        print("[WARNING] No usable chunks created.")
        return

    # -----------------------------
    # 💾 SAVE OUTPUT
    # -----------------------------
    for i, chunk in enumerate(all_chunks):
        chunk["chunk_id"] = i

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"\n[DONE] Created {len(all_chunks)} HIGH-QUALITY chunks.")
    print(f"[OUTPUT] Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    shred_documents()
