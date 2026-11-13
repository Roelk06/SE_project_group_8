# Group 8 - Multi-Modal Spam Detector

## 1. Project Overview

The Multi-Modal Spam Detector is a software system designed
to analyse different types of input and identify potentially
suspicious or spam content.

The system supports:

- SMS and text messages
- Emails
- URLs
- Images (JPG / JPEG)

The application uses a distributed architecture consisting
of an interface server, an orchestrator server and multiple
AI worker servers.

## 2. System Architecture

The system contains three main logical components.

### 2.1 Interface Server

Technology: Streamlit

File: app.py

Responsibilities:

- Provides the graphical user interface.
- Accepts text, URLs and uploaded files (emails as subject + message,
  images as JPG / JPEG, sent inside the request).
- Checks the input (empty input, unsupported image format, image size)
  before sending it.
- Displays classification results.
- Maintains session-based scan history.
- Sends analysis requests to the orchestrator.

The interface does not communicate directly with AI servers.

### 2.2 Orchestrator Server

Technology: Python, FastAPI and Uvicorn

File: services/orchestrator.py

Default address: http://127.0.0.1:8000

Responsibilities:

- Receives requests from the interface.
- Maintains a registry of available AI servers. AI servers send a
  "still alive" message every 10 seconds; a server that has been silent
  for 30 seconds counts as offline.
- Determines which online AI server supports the requested modality.
- Forwards requests to the appropriate AI worker. If a worker cannot be
  reached, the next worker that supports the modality is tried.
- Returns the results to the interface.
- Passes error messages of the AI workers on to the interface
  (for example "Unsupported image format.").
- Returns a clear error when no AI server is online.

### 2.3 AI Worker Servers

Technology: Python, FastAPI and Uvicorn

File: services/ai_server.py

Two AI worker processes are configured.

#### Text AI Worker

Name: ai-text

Default address: http://127.0.0.1:8001

Supported modalities:

- SMS
- Email
- URL

#### Image AI Worker

Name: ai-image

Default address: http://127.0.0.1:8002

Supported modality:

- Image

Both workers run the group's real pipelines and trained models
through input_router.route_input (SMS and email: Linear SVM,
URL: logistic regression, image: ResNet-18). A worker loads the
models first and registers itself afterwards, so the orchestrator
only sends requests to workers that are ready.

After classification the worker stores the result with
input_router.save_result: spam and phishing go to results/spam,
everything else goes to results/ham. The AI server answers with the
standardized result: prediction, label, is_spam, confidence, model
and where the result was stored.

## 3. Communication Architecture

The system uses HTTP communication between its services.

The request flow is:

1. A user submits input through the Streamlit interface.
2. The interface sends an HTTP POST request to the orchestrator.
3. The orchestrator identifies the appropriate AI worker.
4. The orchestrator forwards the request to that worker.
5. The AI worker processes the input.
6. The AI worker returns a classification result.
7. The orchestrator forwards the result to the interface.
8. The interface displays the result to the user.

AI workers register themselves with the orchestrator
when they start and then repeat this every 10 seconds
(this also re-registers them if the orchestrator is restarted).

### Request payload per modality

| Modality | payload |
|---|---|
| sms | text |
| url | text |
| email | {"subject": "...", "message": "..."} |
| image | {"filename": "photo.jpg", "content_b64": "<base64 of the file>"} |

The image is sent inside the request because the AI server is a
separate program and does not share files with the interface.

## 4. Running the System

### Requirements

- Python 3.10 or newer
- pip
- The trained models in models/ and the vectorizers in data/processed/
  (created by the pipelines and training scripts)

### Installation

Install the project dependencies:

    pip install -r requirements.txt

### Starting all services

Run:

    python start_all.py

This starts:

- Orchestrator
- Text AI worker
- Image AI worker
- Streamlit interface

To start everything except the interface:

    python start_all.py --no-ui

Press Ctrl+C to stop the services.

### Individual service addresses

| Service | Address |
|---|---|
| Orchestrator | 127.0.0.1:8000 |
| Text AI worker | 127.0.0.1:8001 |
| Image AI worker | 127.0.0.1:8002 |
| Streamlit | 127.0.0.1:8501 |

## 5. API Endpoints

### Orchestrator

GET /health

Checks whether the orchestrator is running.

GET /services

Returns the registered AI workers and whether each one is online.

POST /register

Registers an AI worker. Workers call it again every 10 seconds
as a "still alive" message.

POST /analyze

Receives an analysis request and forwards it
to the appropriate worker.

### AI Workers

GET /health

Checks whether the AI worker is running.

POST /analyze

Processes a request and returns the classification.

## 6. Deployment

The current implementation runs multiple logical services
on a single computer.

Each service operates as an independent process and
communicates using HTTP.

This allows the architecture to be distributed across
multiple machines in the future by changing service
network addresses and deployment configuration.

## 7. Future Improvements

- Improve error handling and monitoring.
- Introduce authentication between services.
- Add persistent storage for scan history.
- Deploy services to a university server or cloud VM.
- Consider asynchronous task processing using RabbitMQ
  if required by the project.