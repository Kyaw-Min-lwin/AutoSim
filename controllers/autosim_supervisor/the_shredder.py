import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

DOCS_DIR = "./docs"
OUTPUT_FILE = "shredded_rag_chunks.json"


def clean_text(text: str) -> str:
    """
    Cleans the input text by filtering out noise, short strings, and irrelevant content.

    Args:
        text (str): The raw text to be cleaned.

    Returns:
        str: The cleaned text, reconstructed with valid lines.
    """
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        line = line.strip()

        # Filter out empty lines or short noise (less than 40 characters)
        if not line:
            continue
        if len(line) < 40:
            continue

        # Filter out lines containing specific exclusionary keywords
        if "copyright" in line.lower():
            continue
        if "figure" in line.lower():
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


def smart_chunk(text: str, max_words: int = 200) -> list:
    """
    Splits text into manageable chunks based on paragraphs and word counts,
    ensuring each chunk maintains a meaningful size.

    Args:
        text (str): The cleaned text to be chunked.
        max_words (int): The maximum number of words allowed per chunk.

    Returns:
        list: A list of text chunks.
    """
    paragraphs = text.split("\n\n")
    chunks = []

    for p in paragraphs:
        words = p.split()

        # Skip paragraphs that are too short to be meaningful
        if len(words) < 40:
            continue

        # Safely split larger paragraphs into specified max_words increments
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i : i + max_words])

            # Ensure the resulting chunk meets the minimum length requirement
            if len(chunk.split()) < 40:
                continue

            chunks.append(chunk.strip())

    return chunks


def shred_documents():
    """
    Main execution pipeline to process documents from the input directory,
    clean and chunk their contents, and save the structured output to a JSON file.
    """
    print("Starting document processing job...")
    print("-" * 50)

    # Initialize input directory if it does not exist
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(
            f"Notice: Created input directory '{DOCS_DIR}'. Please add your files and restart."
        )
        return

    all_chunks = []

    # Iterate through all files in the designated directory
    for filename in os.listdir(DOCS_DIR):
        filepath = os.path.join(DOCS_DIR, filename)

        # Process PDF files
        if filename.endswith(".pdf"):
            print(f"Processing PDF file: {filename}")
            loader = PyPDFLoader(filepath)
            pages = loader.load()

            for page in pages:
                cleaned = clean_text(page.page_content)
                chunks = smart_chunk(cleaned)

                for c in chunks:
                    all_chunks.append({"source": filename, "type": "pdf", "content": c})

        # Process JSON files (expects pre-cleaned/structured data)
        elif filename.endswith(".json"):
            print(f"Processing JSON file: {filename}")
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
                    print(f"Error processing JSON file '{filename}': {e}")

    # Halt execution if no valid chunks were generated
    if not all_chunks:
        print(
            "Warning: No usable text chunks were extracted from the provided documents."
        )
        return

    # Assign unique identifiers to each chunk for downstream processing
    for i, chunk in enumerate(all_chunks):
        chunk["chunk_id"] = i

    # Write the final payload to the output file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print("-" * 50)
    print(f"Processing complete. Successfully generated {len(all_chunks)} chunks.")
    print(f"Output written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    shred_documents()
