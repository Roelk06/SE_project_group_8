# Multi-Modal Spam Detector

A Software Engineering project that detects spam and suspicious content in:

- SMS messages
- Emails
- URLs
- JPG/JPEG images

## How it works

The application consists of:

- **Streamlit interface**: `services/app.py`
- **Orchestrator**: routes requests to the correct AI service
- **AI services**: process text, URLs, emails, and images
- **Input router**: `input_router.py` loads the appropriate model and stores results in `results/`

The services communicate through HTTP and can be started together with one command.

## Setup

Requirements:

- Python 3.10 or newer
- Trained models in `models/`
- Processed data and vectorizers in `data/processed/`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the application

Run these commands from the root:

```bash
python start_all.py
python start_all.py --no-ui
```

## Run tests

Run all tests:

```bash
python run_tests.py
```

## Project structure

services/
├── app.py              Streamlit interface
├── orchestrator.py     Request routing service
└── ai_server.py        AI worker service

pipelines/              Input preprocessing
training/               Model training scripts
input_router.py         Model prediction and result storage
start_all.py            Starts all application services

tests/                  Unit and integration tests
run_tests.py            Test runner and report generator
pytest.ini              Pytest configuration

data/                   Datasets and processed data
models/                 Trained models
results/                Classified inputs



