from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

app = FastAPI()

class BrainInput(BaseModel):
    log: Dict[str, Any]

@app.get("/")
def home():
    return {"status": "API running"}

@app.post("/brain")
def run_brain(data: BrainInput):
    # TEMP: dummy response for testing
    return {
        "reasoning": "test response",
        "target_parameters": {
            "kp": 1.0,
            "ki": 0.1,
            "kd": 0.05,
            "base_speed": 3.0
        }
    }