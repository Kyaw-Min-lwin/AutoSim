import json
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

DB_DIR = "./chroma_db"

SYSTEM_PROMPT = """You are the AutoSim RAG-Augmented Debugger for a 3D Webots physics simulation.
You do not know what robot you are controlling until you read the logs and the REFERENCE MANUAL.

Analyze the provided JSON failure log. 
Identify the root cause and output ONLY a valid JSON object with:
{{
  "target_parameters": {{
      "exact_actuator_name_from_manifest": number
  }},
  "reasoning": "short explanation based on the reference manual"
}}

Rules:
- Do NOT output anything except JSON.
- FIX THE AGENT, NOT THE ENVIRONMENT.
- You may ONLY target actuator names listed in the 'hardware_manifest.actuators' array.
- CRITICAL: Read the 'mission_objective'. Surviving a collision is NOT enough. You MUST move closer to the target. If you just back up forever, you will fail. Use asymmetric motor velocities to turn towards the objective if necessary.
- CRITICAL: Review the 'failed_attempts' array. It tells you exactly why your previous patches failed (Crash or Cowardice). Learn from them.
- Use the REFERENCE MANUAL below to calculate exact limits, PID adjustments, or physical constraints.

=== REFERENCE MANUAL ===
{rag_context}
"""


def build_llm():
    return ChatOpenAI(
        model="gpt-5-mini",
        temperature=0.2,  # Added slight temp to prevent getting stuck in infinite loops
        model_kwargs={"response_format": {"type": "json_object"}},
    )


def retrieve_context(log_data):
    """Queries ChromaDB based on the specific failure"""
    print("[RAG] Connecting to Vector Database...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    error_type = log_data.get("error_type", "Unknown Error")
    message = log_data.get("message", "")
    actuators = ", ".join(log_data.get("hardware_manifest", {}).get("actuators", []))

    search_query = f"Error: {error_type}. {message}. Hardware involved: {actuators}. How to fix or tune this?"
    print(f"[RAG] Querying vault for: {search_query}")

    docs = vector_db.similarity_search(search_query, k=3)

    context = ""
    for i, doc in enumerate(docs):
        context += (
            f"\n[Doc {i+1} from {doc.metadata.get('source')}]:\n{doc.page_content}\n"
        )

    print(f"[RAG] Retrieved {len(docs)} relevant manual sections.")
    return context


def run_debugger_brain(error_file_path="auto_failure_log.json"):
    print("[BRAIN] Loading failure context...")
    try:
        with open(error_file_path, "r") as f:
            log_data = json.load(f)
    except FileNotFoundError:
        print("[BRAIN] No error log found.")
        return False

    rag_context = retrieve_context(log_data)
    compact_log = json.dumps(log_data, separators=(",", ":"))
    llm = build_llm()

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "Failure log: {failure_log}")]
    )

    chain = prompt | llm

    print("[BRAIN] Analyzing hardware, RAG manual, & mission objective...")

    response = chain.invoke({"rag_context": rag_context, "failure_log": compact_log})

    try:
        result = json.loads(response.content)
        assert "target_parameters" in result
        assert "reasoning" in result
    except Exception as e:
        print("[BRAIN] Invalid response from model:")
        print(response.content)
        return False

    adjustment_command = {
        "action": "update_agent_config",
        "target_parameters": result["target_parameters"],
        "reasoning": result["reasoning"],
    }

    with open("adjustment_command.json", "w") as f:
        json.dump(adjustment_command, f, indent=4)

    print(f"[BRAIN] Patch Generated: {result['reasoning']}")
    return True


if __name__ == "__main__":
    run_debugger_brain()
