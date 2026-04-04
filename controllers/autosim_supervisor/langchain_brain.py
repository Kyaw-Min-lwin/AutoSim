import json
import os
import pickle
import logging
from typing import Dict, Any, List, Tuple
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever  # Required for unpickling
from dotenv import load_dotenv

# Let's make sure we can actually see the system thinking in the terminal
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

load_dotenv()

DB_DIR = "./chroma_db"
BM25_FILE = "bm25_retriever.pkl"
MAX_CHARS_PER_DOC = 800  # Token explosion safety cap to keep the LLM focused

# ==========================================
# 1. CACHED SINGLETONS (Performance Upgrade)
# ==========================================
# By caching these, we avoid the expensive overhead of reconnecting
# to Chroma and unpickling BM25 every single time the drone crashes.
_CHROMA_DB = None
_BM25_RETRIEVER = None
_EMBEDDINGS = None


def get_retrievers():
    """Initializes and caches the retrievers so we don't reload them every tick."""
    global _CHROMA_DB, _BM25_RETRIEVER, _EMBEDDINGS

    if _EMBEDDINGS is None:
        logging.info("Initializing OpenAI Embeddings...")
        _EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-small")

    if _CHROMA_DB is None:
        logging.info("Connecting to Chroma Vector Database...")
        _CHROMA_DB = Chroma(persist_directory=DB_DIR, embedding_function=_EMBEDDINGS)

    if _BM25_RETRIEVER is None:
        logging.info(f"Loading BM25 Sparse Retriever from {BM25_FILE}...")
        try:
            with open(BM25_FILE, "rb") as f:
                _BM25_RETRIEVER = pickle.load(f)
        except FileNotFoundError:
            logging.error(f"'{BM25_FILE}' not found. Run hybrid_db_populator.py first.")
            return None, None

    # We ask Chroma for a few extra docs here so our Ensemble has options
    return _CHROMA_DB.as_retriever(search_kwargs={"k": 4}), _BM25_RETRIEVER


# ==========================================
# 2. PROMPTS
# ==========================================
SYSTEM_PROMPT = """You are the AutoSim RAG-Augmented Autonomy Tuner for a 3D Webots physics simulation.
You no longer control raw motor velocities. You are a Systems Engineer tuning high-level Skills.

Analyze the provided JSON failure log, which includes Episode Summaries and Failure Diagnostics.
Identify the root cause of the failure and output ONLY a valid JSON object with:
{{
  "selected_skill": "DriveToTargetSkill",
  "target_parameters": {{
      "kp": float (Proportional gain for heading adjustment),
      "ki": float (Integral gain for heading adjustment),
      "kd": float (Derivative damping for heading adjustment),
      "base_speed": float (Forward velocity)
  }},
  "reasoning": "short explanation of why you tuned these specific parameters based on the episode summary"
}}

Rules:
- Do NOT output anything except valid JSON.
- Currently, the only available skill is "DriveToTargetSkill".
- CRITICAL: Read the 'error_type' and 'message' from the diagnostic engine.
    - If 'DynamicInstability' (wobble) or 'Thrashing', you MUST reduce 'kp' and increase 'kd' (damping).
    - If 'SevereDrift', your 'kp' might be too low to overcome momentum, or 'base_speed' is too fast to turn cleanly.
    - If 'KineticStagnation', your 'base_speed' is too low to overcome friction, or you are stuck.
- Review the 'failed_attempts' array. It tells you exactly why your previous PID tuning failed. Learn from it.
- Use the REFERENCE MANUAL below to calculate exact limits, understand the physical constraints, or read up on PID tuning theory.

=== REFERENCE MANUAL ===
{rag_context}
"""

# The HyDE prompt tricks the LLM into writing the perfect answer, which we then use as our search query!
HYDE_PROMPT = """You are an expert professor of robotics and control theory.
A differential drive robot in a physics simulation just experienced a failure.
Error Type: {error_type}
Message: {message}
Prior failed attempts at fixing this: {failed_context}

Do NOT solve the problem. Instead, write a hypothetical, highly technical textbook excerpt that explains the control theory concepts needed to fix this specific failure using PID parameters (kp, ki, kd) and base velocity. Write it as if it is a section from a robotics manual.
"""


# ==========================================
# 3. ADVANCED RETRIEVAL LOGIC
# ==========================================
def generate_hyde_document(
    log_data: Dict[str, Any], llm: ChatOpenAI
) -> Tuple[str, str]:
    """Generates a hypothetical ideal document to use as the embedding search query."""
    error_type = log_data.get("error_type", "Unknown Error")
    message = log_data.get("message", "No message provided")

    # Extract prior failures to make the query control-aware
    failed_attempts = log_data.get("failed_attempts", [])
    failed_context = "None."
    if failed_attempts:
        failed_context = "; ".join(
            [
                f"Tried {fa.get('parameters', 'unknown')} but resulted in {fa.get('reason', 'failure')}"
                for fa in failed_attempts
            ]
        )

    prompt = ChatPromptTemplate.from_messages([("user", HYDE_PROMPT)])
    chain = prompt | llm

    logging.info("Generating HyDE (Hypothetical Document Embedding)...")
    hypo_doc = chain.invoke(
        {"error_type": error_type, "message": message, "failed_context": failed_context}
    ).content

    logging.info(f"HyDE Output Generated (Snippet): {hypo_doc[:100]}...")
    return hypo_doc, error_type


def rerank_documents(docs: List[Document], top_k: int = 3) -> List[Document]:
    """
    Lightweight heuristic filter to ensure we only pass the best chunks to the main LLM
    and strictly enforce token limits.
    """
    logging.info(
        f"Filtering {len(docs)} retrieved documents down to the top {top_k}..."
    )

    final_docs = docs[:top_k]
    for doc in final_docs:
        # Strict Token Capping! If a chunk is too massive, we chop it.
        # This prevents the final GPT prompt from exploding and confusing the model.
        if len(doc.page_content) > MAX_CHARS_PER_DOC:
            doc.page_content = (
                doc.page_content[:MAX_CHARS_PER_DOC] + "... [TRUNCATED FOR EFFICIENCY]"
            )

        logging.info(
            f"Retained Doc Source: {doc.metadata.get('source', 'Unknown')} | Type: {doc.metadata.get('type', 'Unknown')}"
        )

    return final_docs


def retrieve_context(log_data: Dict[str, Any]) -> str:
    """
    Executes the advanced HyDE + Dynamic Ensemble pipeline.
    """
    chroma_retriever, bm25_retriever = get_retrievers()
    if not chroma_retriever or not bm25_retriever:
        return "Warning: Could not load retrievers."

    # Use a faster, cheaper text-only LLM for our retrieval sub-tasks
    retrieval_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    # 1. Generate the HyDE query
    hyde_query, error_type = generate_hyde_document(log_data, retrieval_llm)

    # 2. Dynamic Weighting based on error state
    if "Unknown" in error_type or "General" in error_type:
        weights = [
            0.3,
            0.7,
        ]  # 30% Keyword, 70% Semantic (We need theory for unknown stuff)
        logging.info("Dynamic Weighting: Leaning Dense (Semantic) for Unknown Error.")
    else:
        weights = [
            0.6,
            0.4,
        ]  # 60% Keyword, 40% Semantic (Specific errors need specific terms!)
        logging.info(
            "Dynamic Weighting: Leaning Sparse (Keyword) for Known Control Error."
        )

    # Apply limits to BM25 retriever before ensemble
    bm25_retriever.k = 4

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever], weights=weights
    )

    logging.info("Executing Hybrid Search using HyDE query...")
    raw_docs = ensemble_retriever.invoke(hyde_query)

    # 3. Filter and Cap Tokens
    refined_docs = rerank_documents(raw_docs, top_k=3)

    context = ""
    for i, doc in enumerate(refined_docs):
        context += f"\n[Document {i+1} source: {doc.metadata.get('source', 'Unknown')}]:\n{doc.page_content}\n"

    logging.info("Context compiled successfully.")
    return context


# ==========================================
# 4. MAIN EXECUTION (The Systems Engineer)
# ==========================================
def run_debugger_brain(error_file_path: str = "auto_failure_log.json") -> bool:
    """Main execution flow to act as the Systems Engineer."""
    print("=" * 60)
    print("🧠 INITIATING AUTO-SIM RAG DEBUGGER (HyDE ENHANCED)")
    print("=" * 60)

    try:
        with open(error_file_path, "r") as f:
            log_data = json.load(f)
    except FileNotFoundError:
        logging.error(f"No error log found at '{error_file_path}'. Terminating.")
        return False

    # Fetch hyper-refined context using our beast of a retrieval pipeline
    rag_context = retrieve_context(log_data)
    compact_log = json.dumps(log_data, separators=(",", ":"))

    # The main logic LLM, strictly bound to JSON output
    main_llm = ChatOpenAI(
        model="gpt-5-mini",
        temperature=0.1,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    formatted_system_prompt = SYSTEM_PROMPT.format(rag_context=rag_context)
    prompt = ChatPromptTemplate.from_messages(
        [("system", formatted_system_prompt), ("human", "Failure log: {failure_log}")]
    )

    chain = prompt | main_llm

    logging.info(
        "Synthesizing telemetry, analyzing physics, and calculating optimal PID patch..."
    )
    response = chain.invoke({"failure_log": compact_log})

    try:
        # Safely strip Markdown formatting that LLMs love to sneak in, avoiding syntax errors
        clean_content = response.content.strip()
        if clean_content.startswith("```json"):
            clean_content = clean_content[7:]
        elif clean_content.startswith("```"):
            clean_content = clean_content[3:]

        if clean_content.endswith("```"):
            clean_content = clean_content[:-3]

        result = json.loads(clean_content.strip())
        assert "target_parameters" in result
        assert "selected_skill" in result
        assert "reasoning" in result

    except Exception as e:
        logging.error(f"Malformed LLM Response: {e}")
        # Explicit error state to prevent Webots infinite loops!
        error_command = {
            "error": True,
            "reasoning": "LLM output invalid. Retry required.",
        }
        with open("adjustment_command.json", "w") as f:
            json.dump(error_command, f, indent=4)
        return False

    with open("adjustment_command.json", "w") as f:
        json.dump(result, f, indent=4)

    print("-" * 60)
    print(f"✅ SKILL PATCH GENERATED")
    print(f"Reasoning: {result['reasoning']}")
    print(f"Parameters: {result['target_parameters']}")
    print("Command saved to 'adjustment_command.json'. Supervisor ready for restart.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    run_debugger_brain()
