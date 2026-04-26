from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

# Import the processing function from your LangChain file
from langchain_brain import run_debugger_brain

app = FastAPI()


class BrainInput(BaseModel):
    log: Dict[str, Any]


@app.get("/")
def home():
    return {"status": "Brain API running"}


@app.post("/brain")
def run_brain(data: BrainInput):
    """
    Receives the failure log, passes it to the LangChain RAG pipeline,
    and returns the tuned PID parameters.
    """
    try:
        # Call the LangChain logic directly with the dictionary
        command_output = run_debugger_brain(data.log)

        # If the LLM failed to parse or crashed, raise a 500 error
        # so your requests.post().raise_for_status() catches it properly
        if command_output.get("error"):
            raise HTTPException(status_code=500, detail=command_output.get("reasoning"))

        return command_output

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
