# Multi-Modal Spam Detector

A Software Engineering project that detects spam and suspicious content in:

- SMS messages
- Emails
- URLs
- JPG/JPEG images

## How it works

The application is split into separate services that talk to each other over HTTP:

- **Streamlit interface** (`services/app.py`): takes the user's input, checks it
  (empty input, image format and size) and sends it to the orchestrator. It never
  calls the AI models directly.
- **Orchestrator** (`services/orchestrator.py`): keeps a list of the AI servers that
  are online and forwards each request to a server that supports that input type.
  If a server cannot be reached, the next one is tried.
- **AI servers** (`services/ai_server.py`): run the pipelines and trained models
  through `input_router.py` and store each result in `results/spam` or `results/ham`.
  Two are started: `ai-text` (SMS, email, URL) and `ai-image` (images).
  Every AI server registers itself with the orchestrator and repeats that every
  10 seconds; a server that has been silent for 30 seconds counts as offline.

```
User ──> Interface ──> Orchestrator ──> ai-text  (SMS, email, URL)
           :8501          :8000     └──> ai-image (images)
                                          :8001 / :8002
```

All services run on this computer (127.0.0.1). Because they only communicate
over HTTP, each one could be moved to another machine by changing its address.

## Setup

The datasets and trained models are not in git (they are too large), so every
group member sets them up once on their own computer.

### 1. Install the packages

Python 3.10 or newer is needed.

```bash
pip install -r requirements.txt
```

### 2. Put the datasets in `data/raw/`

The pipelines look for the files at these exact places:

```
data/raw/
├── SMS/SMSSpamCollection            SMS Spam Collection (UCI), the file without extension
├── emails/enron_spam_data.csv       Enron spam dataset
├── urls/phishing_site_urls.csv      Phishing site URLs (Kaggle)
└── images/
    ├── ham/                         normal images (NaturalImages in the download)
    └── spam/                        spam images (SpamImages in the download)
```

Put the image files directly inside `ham/` and `spam/`, not in another subfolder.

### 3. Run the pipelines

This creates `data/processed/` (train/validation/test splits and the vectorizers):

```bash
python pipelines/sms_pipeline.py
python pipelines/email_pipeline.py
python pipelines/url_pipeline.py
python pipelines/images_pipeline.py
```

### 4. Train the models

This creates the trained models in `models/`. The image model takes the longest
(about 10 minutes on a laptop without a graphics card).

```bash
python training/sms_model_v1.1.py
python training/email_model_v1.1.py
python training/url_model_v1.2.5.py
python training/images_model_v1.2.py
```

Only run the scripts in `training/`. The ones in `training/archive/` are old
versions that save to the same file names and would overwrite the new models.

After changing a pipeline, run that pipeline and its training script again, and
restart the application.

## Run the application

Run these commands from the project folder:

```bash
python start_all.py            # everything, then open http://localhost:8501
python start_all.py --no-ui    # everything except the interface
```

Press Ctrl+C to stop all services.

## Run tests

```bash
python run_tests.py                 # all tests
python run_tests.py unit            # only the unit tests
python run_tests.py integration     # only the integration tests
```

The results are also saved to `tests/test_report.md`.

## Models

| Input | Model | Notes |
|---|---|---|
| SMS | Linear SVM | TF-IDF of words + extra features (capitals, `!`, `?`, phone numbers, money amounts, links, stretched words like "FREEEEE") |
| Email | Linear SVM | TF-IDF of subject + message |
| URL | Logistic regression | TF-IDF of character groups; `http(s)://` and `www.` are removed first. Well-known domains (list in `pipelines/url_pipeline.py`) are always legitimate |
| Image | ResNet-18 | Pretrained on ImageNet, fine-tuned on spam images |

When the confidence is below the threshold in the sidebar (default 70%), the
interface shows the result as **uncertain**.

### Known limitations

- The SMS dataset contains British text messages from 2012, so modern spam
  ("free money", crypto, parcel delivery) is not always recognised.
- The URL dataset contains almost no links with `https://`, and the model only
  reads the characters of a URL, not who owns the website. Unknown but harmless
  websites are therefore sometimes marked as phishing (usually with low confidence).
- Phishing pages hosted on a trusted domain (for example a fake Google Form) are
  let through by the trusted domain list.

## Project structure

```
services/
├── app.py              Streamlit interface
├── orchestrator.py     Request routing service
└── ai_server.py        AI worker service

pipelines/              Input preprocessing
training/               Model training scripts
└── archive/            Old versions, kept for reference (do not run)
input_router.py         Model prediction and result storage
start_all.py            Starts all application services

tests/                  Unit and integration tests
run_tests.py            Test runner and report generator
pytest.ini              Pytest configuration

data/                   Datasets and processed data (not in git)
models/                 Trained models (not in git)
results/                Classified inputs (not in git)
```
