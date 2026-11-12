import requests

from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Spam Detector Interface")
ORCHESTRATOR_URL = "http://127.0.0.1:8000"

class AnalysisRequest(BaseModel):
    modality: str
    payload: Any
    source_name: str = "Direct Input"

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "interface"}

@app.post("/analyze")
def analyze(request: AnalysisRequest):
    try:
        response = requests.post(
            f"{ORCHESTRATOR_URL}/analyze",
            json={
                "modality": request.modality,
                "payload": request.payload,
                "source_name": request.source_name
            },
            timeout=70)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="The orchestrator is not available")

    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="The request timed out")

    except requests.exceptions.HTTPError as error:
        try:
            detail = error.response.json().get("detail", str(error))
        except ValueError:
            detail = str(error)
        raise HTTPException(
            status_code=error.response.status_code,
            detail=detail)

    except requests.exceptions.RequestException as error:
        raise HTTPException(
            status_code=502,
            detail=f"Communication with orchestrator failed: {error}")
