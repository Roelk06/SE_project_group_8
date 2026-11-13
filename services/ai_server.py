"""
AI worker service for Group 8's Multi-Modal Spam Detector.

The same program can start different AI workers:
- ai-text: SMS, email and URL
- ai-image: image processing

Each worker runs the project's real pipelines and trained models
(input_router.route_input) and sorts the result into results/spam or
results/ham (input_router.save_result).
"""

import argparse
import base64
import binascii
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# input_router.py and the pipelines folder are in the project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


app = FastAPI(title="Spam Detector AI Worker")

SERVER_NAME = ""
SUPPORTED_TYPES = []
ORCHESTRATOR_URL = ""
PORT = 8001

# How often the worker tells the orchestrator that it is still alive.
HEARTBEAT_SECONDS = 10

# The input_router module. It is loaded once when the server starts,
# because importing it loads torch and the pipelines (slow).
ROUTER = None

MODEL_NAMES = {
    "sms": "sms_linear_svm",
    "email": "email_linear_svm",
    "url": "url_logistic_regression",
    "image": "image_resnet18"
}

# The URL model says phishing / legitimate, the others spam / ham.
SPAM_LABELS = {"spam", "phishing"}


class AnalysisRequest(BaseModel):
    modality: str
    # sms / url: text
    # email: {"subject": ..., "message": ...}
    # image: {"filename": ..., "content_b64": ...}
    payload: Any
    source_name: str = "Direct Input"


def get_router():
    global ROUTER

    if ROUTER is None:
        import input_router
        ROUTER = input_router

    return ROUTER


def build_router_input(modality: str, payload, workdir: str):
    """Turns the request payload into what input_router.route_input expects."""

    try:
        if modality in ("sms", "url"):
            return payload

        if modality == "email":
            return {
                "subject": payload["subject"],
                "message": payload["message"]
            }

        if modality == "image":

            try:
                raw = base64.b64decode(payload["content_b64"], validate=True)
            except (binascii.Error, ValueError):
                raise ValueError("The image data could not be read.")

            # The router reads images from a file: keep a temporary copy.
            filename = Path(payload.get("filename") or "upload.jpg").name
            path = Path(workdir) / f"image{Path(filename).suffix.lower()}"
            path.write_bytes(raw)

            return path

    except (KeyError, TypeError, AttributeError):
        raise ValueError(f"The {modality} request has the wrong format.")

    raise ValueError(f"Unsupported modality: {modality}")


def classify(modality: str, payload) -> dict:
    """Runs the real pipeline + model and stores the result in results/spam or results/ham."""

    router = get_router()

    with tempfile.TemporaryDirectory(prefix="ai_server_") as workdir:

        data = build_router_input(modality, payload, workdir)

        label, confidence = router.route_input(modality, data)

        stored = router.save_result(
            modality,
            data,
            label,
            confidence=confidence
        )

    is_spam = label in SPAM_LABELS

    return {
        "prediction": label.capitalize(),
        "label": label,
        "is_spam": is_spam,
        "confidence": confidence,
        "model": MODEL_NAMES[modality],
        "stored_in": f"results/{stored['label']}"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "server": SERVER_NAME,
        "supported_types": SUPPORTED_TYPES
    }


@app.post("/analyze")
def analyze(request: AnalysisRequest):

    modality = request.modality.lower()

    if modality not in SUPPORTED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"{SERVER_NAME} does not support {modality}"
        )

    start_time = time.perf_counter()

    try:
        result = classify(modality, request.payload)

    except ValueError as error:
        # Input problem (unsupported image format, invalid URL, ...):
        # the message is meant for the user.
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                f"{SERVER_NAME} could not process this input "
                f"({type(error).__name__}: {error})"
            )
        )

    latency_ms = round(
        (time.perf_counter() - start_time) * 1000,
        1
    )

    return {
        "source": request.source_name,
        "modality": modality,
        **result,
        "latency_ms": latency_ms,
        "status": "success"
    }


def build_registration():

    return {
        "name": SERVER_NAME,
        "url": f"http://127.0.0.1:{PORT}",
        "types": SUPPORTED_TYPES
    }


def register_with_orchestrator():

    registration = build_registration()

    # Retry because the orchestrator may still be starting.
    for attempt in range(30):

        try:
            response = requests.post(
                f"{ORCHESTRATOR_URL}/register",
                json=registration,
                timeout=3
            )

            response.raise_for_status()

            print(f"{SERVER_NAME} registered successfully.")
            return

        except requests.exceptions.RequestException:

            time.sleep(1)

    raise RuntimeError(
        "Could not register AI server with orchestrator."
    )


def heartbeat_loop():
    """
    Tells the orchestrator regularly that this server is still alive.
    This also registers the server again if the orchestrator was restarted.
    """

    while True:

        time.sleep(HEARTBEAT_SECONDS)

        try:
            requests.post(
                f"{ORCHESTRATOR_URL}/register",
                json=build_registration(),
                timeout=3
            )
        except requests.exceptions.RequestException:
            pass


def main():

    global SERVER_NAME
    global SUPPORTED_TYPES
    global ORCHESTRATOR_URL
    global PORT

    parser = argparse.ArgumentParser()

    parser.add_argument("--name", required=True)
    parser.add_argument("--types", nargs="+", required=True)

    parser.add_argument(
        "--orchestrator",
        default="http://127.0.0.1:8000"
    )

    parser.add_argument("--port", type=int, default=None)

    args = parser.parse_args()

    SERVER_NAME = args.name
    SUPPORTED_TYPES = [item.lower() for item in args.types]
    ORCHESTRATOR_URL = args.orchestrator.rstrip("/")

    if args.port is not None:
        PORT = args.port
    elif SERVER_NAME == "ai-text":
        PORT = 8001
    elif SERVER_NAME == "ai-image":
        PORT = 8002
    else:
        raise ValueError(
            "Specify --port for an unrecognised AI server name."
        )

    # Load the models first, then register: the orchestrator
    # only sends requests to servers that are ready.
    @app.on_event("startup")
    def startup():
        print(f"{SERVER_NAME}: loading the models and pipelines...")
        get_router()
        register_with_orchestrator()
        threading.Thread(target=heartbeat_loop, daemon=True).start()

    print(f"Starting {SERVER_NAME} on port {PORT}")

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=PORT,
        log_level="info"
    )


if __name__ == "__main__":
    main()
