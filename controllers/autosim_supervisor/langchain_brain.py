import json
import os
from typing import Dict, Any
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


def build_llm() -> ChatOpenAI:
    """
    Initializes and returns the language model configuration for the debugger.

    Returns:
        ChatOpenAI: The configured OpenAI chat model.
    """
    return ChatOpenAI(
        model="gpt-5-mini",
        temperature=0.2,  # Added slight temp to prevent getting stuck in infinite loops
        model_kwargs={"response_format": {"type": "json_object"}},
    )


def retrieve_context(log_data: Dict[str, Any]) -> str:
    """
    Queries the Chroma vector database to retrieve relevant hardware manual sections
    based on the specific failure log.

    Args:
        log_data (Dict[str, Any]): The parsed JSON data from the failure log.

    Returns:
        str: Formatted context strings containing the retrieved manual sections.
    """
    print("Connecting to Vector Database...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    error_type = log_data.get("error_type", "Unknown Error")
    message = log_data.get("message", "")
    actuators = ", ".join(log_data.get("hardware_manifest", {}).get("actuators", []))

    search_query = f"Error: {error_type}. {message}. Hardware involved: {actuators}. How to fix or tune this?"
    print(f"Executing semantic search for: '{search_query}'")

    docs = vector_db.similarity_search(search_query, k=3)

    context = ""
    for i, doc in enumerate(docs):
        context += f"\n[Document {i+1} source: {doc.metadata.get('source')}]:\n{doc.page_content}\n"

    print(f"Successfully retrieved {len(docs)} relevant manual sections.")
    return context


def run_debugger_brain(error_file_path: str = "auto_failure_log.json") -> bool:
    """
    Main execution flow to analyze a failure log, retrieve relevant context via RAG,
    and prompt the LLM to generate an adjustment command to fix the simulation agent.

    Args:
        error_file_path (str): The file path to the JSON failure log.

    Returns:
        bool: True if a patch was successfully generated, False otherwise.
    """
    print("Starting automated debugging analysis...")
    print("-" * 50)
    print(f"Loading failure context from '{error_file_path}'...")

    try:
        with open(error_file_path, "r") as f:
            log_data = json.load(f)
    except FileNotFoundError:
        print("Error: No error log found. Terminating process.")
        return False

    # Fetch context and compact the log data for token efficiency
    rag_context = retrieve_context(log_data)
    compact_log = json.dumps(log_data, separators=(",", ":"))

    llm = build_llm()
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "Failure log: {failure_log}")]
    )

    chain = prompt | llm

    print(
        "Analyzing hardware specifications, manual context, and mission objectives..."
    )
    response = chain.invoke({"rag_context": rag_context, "failure_log": compact_log})

    try:
        result = json.loads(response.content)
        assert "target_parameters" in result
        assert "reasoning" in result
    except Exception as e:
        print("Error: Invalid or malformed response received from the language model.")
        print(f"Raw output:\n{response.content}")
        return False

    adjustment_command = {
        "action": "update_agent_config",
        "target_parameters": result["target_parameters"],
        "reasoning": result["reasoning"],
    }

    with open("adjustment_command.json", "w") as f:
        json.dump(adjustment_command, f, indent=4)

    print("-" * 50)
    print(f"Patch Generated Successfully: {result['reasoning']}")
    print("Adjustment command saved to 'adjustment_command.json'.")
    return True


if __name__ == "__main__":
    run_debugger_brain()
