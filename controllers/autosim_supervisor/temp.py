from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

DB_DIR = "./chroma_db"


def test_retriever():
    print(">>> BOOTING VECTOR DATABASE...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Load the existing database (we don't pass documents this time)
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    # The Test Query
    query = "How do I fix steady-state error in a drone?"
    print(f"\n[QUERY] '{query}'")

    # Retrieve the top 2 most mathematically similar chunks
    results = vector_db.similarity_search(query, k=2)

    print("\n[RESULTS FOUND]")
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} (Source: {doc.metadata.get('source')}) ---")
        print(doc.page_content)


if __name__ == "__main__":
    test_retriever()
