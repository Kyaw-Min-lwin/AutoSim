import os
import json
import subprocess
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

DOCS_DIR = "./docs"
OUTPUT_FILE = "shredded_rag_chunks.json"
MARKER_OUTPUT_DIR = "./marker_temp"


def extract_markdown_with_marker(filepath: str, filename: str) -> str:
    print(f"  -> Running Marker OCR/Vision pipeline on {filename}...")
    os.makedirs(MARKER_OUTPUT_DIR, exist_ok=True)
    base_name = os.path.splitext(filename)[0]
    output_md_path = os.path.join(MARKER_OUTPUT_DIR, base_name, f"{base_name}.md")

    command = [
        "marker_single",
        filepath,
        MARKER_OUTPUT_DIR,
        "--batch_multiplier",
        "1",
        "--extract_images",
        "false",
    ]
    try:
        subprocess.run(
            command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
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
    if not markdown_text.strip():
        return []

    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    md_header_splits = markdown_splitter.split_text(markdown_text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=150, separators=["\n\n", "\n", " ", ""]
    )
    final_chunks = text_splitter.split_documents(md_header_splits)

    return [
        {"content": chunk.page_content, "metadata": chunk.metadata}
        for chunk in final_chunks
    ]


def shred_documents():
    print("Starting document processing job... (VLM + Markdown Edition)")
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
        file_ext = filename.lower()

        if file_ext.endswith(".pdf"):
            print(f"Processing PDF file: {filename}")
            markdown_text = extract_markdown_with_marker(filepath, filename)
            chunks = structural_chunk(markdown_text)
            for c in chunks:
                all_chunks.append(
                    {
                        "source": filename,
                        "type": "pdf",
                        "headers": c["metadata"],
                        "content": c["content"],
                    }
                )

        elif file_ext.endswith(".json"):
            print(f"Processing JSON file: {filename}")
            with open(filepath, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    for item in data:
                        content = item.get("content", "")
                        source = item.get("source", filename)
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

        elif file_ext.endswith(".md"):
            print(f"Processing Markdown file: {filename}")
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    markdown_text = f.read()
                chunks = structural_chunk(markdown_text)
                for c in chunks:
                    all_chunks.append(
                        {
                            "source": filename,
                            "type": "md",
                            "headers": c["metadata"],
                            "content": c["content"],
                        }
                    )
            except Exception as e:
                print(f"Error processing Markdown file '{filename}': {e}")

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
