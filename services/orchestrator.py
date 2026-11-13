"""
Orchestrator service for Group 8's Multi-Modal Spam Detector.

Receives requests from the interface and forwards them
to the appropriate registered AI server.
"""

from typing import Any, Dict, List
import time

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI(title="Spam Detector Orchestrator")

# An AI server counts as online if it registered (or sent its
# regular "still alive" message) within this many seconds.
SERVER_TIMEOUT_SECONDS = 30

SUPPORTED_TYPES = {"sms", "email", "url", "image"}

# Registry of AI servers.
# Example:
# {
#   "ai-text": {
#       "url": "http://127.0.0.1:8001",
#       "types": ["sms", "email", "url"],
#       "last_seen": 1700000000.0
#   }
# }
AI_SERVERS: Dict[str, dict] = {}


class ServerRegistration(BaseModel):
    name: str
    url: str
    types: List[str]


class AnalysisRequest(BaseModel):
    modality: str
    # sms / url: text
    # email: {"subject": ..., "message": ...}
    # image: {"filename": ..., "content_b64": ...}
    payload: Any
    source_name: str = "Direct Input"


def is_online(server: dict) -> bool:
    return time.time() - server["last_seen"] <= SERVER_TIMEOUT_SECONDS


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "orchestrator"
    }


@app.get("/services")
def list_services():
    """Show all registered AI servers and whether they are online."""
    return {
        "servers": {
            name: {**server, "online": is_online(server)}
            for name, server in AI_SERVERS.items()
        }
    }


@app.post("/register")
def register_server(registration: ServerRegistration):
    """
    Register an AI server with the orchestrator.

    AI servers call this again every few seconds, which also
    registers them again if the orchestrator was restarted.
    """

    if registration.name not in AI_SERVERS:
        print(f"Registered AI server: {registration.name}")

    AI_SERVERS[registration.name] = {
        "url": registration.url.rstrip("/"),
        "types": [item.lower() for item in registration.types],
        "last_seen": time.time()
    }

    return {
        "status": "registered",
        "server": registration.name
    }


@app.post("/analyze")
def analyze(request: AnalysisRequest):
    """
    Receive a request from the interface,
    select the correct AI server,
    and forward the request.
    """

    modality = request.modality.lower()

    if modality not in SUPPORTED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported input type: {modality}"
        )

    # All online servers that support this modality.
    candidates = [
        (name, server)
        for name, server in AI_SERVERS.items()
        if modality in server["types"] and is_online(server)
    ]

    if not candidates:
        raise HTTPException(
            status_code=503,
            detail=f"No AI server is online for modality: {modality}"
        )

    # Try the servers one by one: if one cannot be reached, use the next.
    for server_name, server_info in candidates:

        try:
            response = requests.post(
                f"{server_info['url']}/analyze",
                json={
                    "modality": modality,
                    "payload": request.payload,
                    "source_name": request.source_name
                },
                timeout=60
            )

            response.raise_for_status()

            result = response.json()

            # Add information about which server processed the request.
            result["ai_server"] = server_name

            return result

        except requests.exceptions.ConnectionError:
            # This server is down: try the next one.
            continue

        except requests.exceptions.Timeout:
            raise HTTPException(
                status_code=504,
                detail=f"AI server {server_name} timed out"
            )

        except requests.exceptions.HTTPError as error:
            # The AI server answered with an error (for example
            # "Unsupported image format."): pass its message on.
            try:
                detail = error.response.json().get("detail", str(error))
            except ValueError:
                detail = str(error)

            status_code = error.response.status_code
            raise HTTPException(
                status_code=status_code if 400 <= status_code < 500 else 502,
                detail=detail
            )

        except requests.exceptions.RequestException as error:
            raise HTTPException(
                status_code=502,
                detail=f"Communication with AI server failed: {error}"
            )

    raise HTTPException(
        status_code=502,
        detail=f"No AI server for {modality} could be reached."
    )
