import os
import json
import subprocess
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# Directory Setup
DOCS_DIR = "./docs"
OUTPUT_FILE = "shredded_rag_chunks.json"
MARKER_OUTPUT_DIR = "./marker_temp"  # Temp dir for Marker's markdown output


def extract_markdown_with_marker(filepath: str, filename: str) -> str:
    """
    Uses the open-source `marker-pdf` CLI to extract Markdown and LaTeX equations.
    """
    print(f"  -> Running Marker OCR/Vision pipeline on {filename}...")

    # Ensure temp directory exists
    os.makedirs(MARKER_OUTPUT_DIR, exist_ok=True)

    # Marker creates a folder named after the file (minus extension)
    base_name = os.path.splitext(filename)[0]
    output_md_path = os.path.join(MARKER_OUTPUT_DIR, base_name, f"{base_name}.md")

    # Command to run marker (uses your GPU/CPU to run deep learning OCR)
    # We call it via subprocess to keep our main Python memory clean
    command = [
        "marker_single",
        filepath,
        MARKER_OUTPUT_DIR,
        "--batch_multiplier",
        "1",  # Adjust based on your VRAM
        "--extract_images",
        "false",  # We just want text/math for the RAG
    ]

    try:
        # Run the VLM pipeline
        subprocess.run(
            command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        # Read the generated Markdown
        if os.path.exists(output_md_path):
            with open(output_md_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            print(f"  -> Warning: Marker failed to output markdown for {filename}")
            return ""

    except subprocess.CalledProcessError as e:
        print(f"  -> Error running Marker on {filename}: {e}")
        return ""


def structural_chunk(markdown_text: str) -> list:
    """
    Splits text semantically based on Markdown headers, keeping context and math intact.
    Falls back to RecursiveCharacter splitting for ridiculously long sections.
    """
    if not markdown_text.strip():
        return []

    # 1. Define the headers we want to split on
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    # Initialize the structural splitter
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,  # Keep the headers in the text so the LLM knows what it's reading
    )

    # Chunk by structure!
    md_header_splits = markdown_splitter.split_text(markdown_text)

    # 2. Safety Net: If a specific section (e.g., a massive math derivation) is STILL
    # too large for an embedding model, we gently split it by character, prioritizing paragraphs.
    chunk_size = 1500
    chunk_overlap = 150
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            " ",
            "",
        ],  # Will not split inside a paragraph unless absolutely necessary
    )

    final_chunks = text_splitter.split_documents(md_header_splits)

    # Return as list of string contents with their metadata (which headers they belong to)
    return [
        {
            "content": chunk.page_content,
            "metadata": chunk.metadata,  # Contains dict like {"Header 1": "Introduction"}
        }
        for chunk in final_chunks
    ]


def shred_documents():
    """
    Main execution pipeline.
    """
    print("Starting document processing job... (VLM Edition)")
    print("-" * 50)

    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(
            f"Notice: Created input directory '{DOCS_DIR}'. Please add your files and restart."
        )
        return

    all_chunks = []

    for filename in os.listdir(DOCS_DIR):
        filepath = os.path.join(DOCS_DIR, filename)

        # Process PDF files with Marker
        if filename.endswith(".pdf"):
            print(f"Processing PDF file: {filename}")

            # Step 1: Use Vision/OCR models to get pure Markdown + LaTeX
            markdown_text = extract_markdown_with_marker(filepath, filename)

            # Step 2: Chunk structurally
            structural_chunks = structural_chunk(markdown_text)

            for c in structural_chunks:
                all_chunks.append(
                    {
                        "source": filename,
                        "type": "pdf",
                        "headers": c["metadata"],
                        "content": c["content"],
                    }
                )

        # Process JSON files
        elif filename.endswith(".json"):
            print(f"Processing JSON file: {filename}")
            with open(filepath, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    for item in data:
                        content = item.get("content", "")
                        source = item.get("source", filename)

                        # If the JSON contains markdown, structurally chunk it!
                        chunks = structural_chunk(content)

                        for c in chunks:
                            all_chunks.append(
                                {
                                    "source": source,
                                    "type": "json",
                                    "headers": c["metadata"],
                                    "content": c["content"],
                                }
                            )
                except Exception as e:
                    print(f"Error processing JSON file '{filename}': {e}")

    if not all_chunks:
        print(
            "Warning: No usable text chunks were extracted from the provided documents."
        )
        return

    for i, chunk in enumerate(all_chunks):
        chunk["chunk_id"] = i

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print("-" * 50)
    print(
        f"Processing complete. Successfully generated {len(all_chunks)} semantically intact chunks."
    )
    print(f"Output written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    shred_documents()
