# SE_project_group_8
A school project for the course: Software Engineering for CSAI.


Please check with the person who wrote the code if you remove or change it.

## Project structure

```
├── app.py              Streamlit interface
├── start_all.py        starts the whole system (orchestrator, AI servers, interface)
├── services/           orchestrator and AI server
├── input_router.py     gets the model's prediction, puts spam in results/spam and the rest in results/ham
├── pipelines/          preprocessing for each input type
├── training/           training scripts for each model (latest versions)
│   └── archive/        original versions, kept for reference and not meant to be run
├── tests/              unit and integration tests (see tests/notes.txt)
├── run_tests.py        runs the tests and writes tests/test_report.md
├── pytest.ini          pytest settings
├── data/               datasets (not in git)
├── models/             saved models (not in git)
└── results/            sorted inputs (not in git, created when first used)
```

## Setup

The datasets and trained models are not in git, so every group member has to
set them up once on their own computer.

### 1. Install the packages

```
pip install -r requirements.txt
```

### 2. Download the datasets and put them in `data/raw/`

The pipelines look for the files at these exact places (create the folders yourself):

```
data/raw/
├── SMS/SMSSpamCollection            SMS Spam Collection (UCI), the file without extension
├── emails/enron_spam_data.csv       Enron spam dataset
├── urls/phishing_site_urls.csv      Phishing site URLs (Kaggle)
└── images/
    ├── ham/                         all normal images (NaturalImages in the download)
    └── spam/                        all spam images (SpamImages in the download)
```

Put the image files directly inside `ham/` and `spam/`, not in another subfolder.

### 3. Run the pipelines

This creates `data/processed/` (train/val/test splits and the vectorizers):

```
python pipelines/sms_pipeline.py
python pipelines/email_pipeline.py
python pipelines/url_pipeline.py
python pipelines/images_pipeline.py
```

### 4. Train the models

This creates the models in `models/`. The image model takes the longest,
especially on a computer without a graphics card.

```
python training/sms_model_v1.1.py
python training/email_model_v1.1.py
python training/url_model_v1.2.5.py
python training/images_model_v1.2.py
```

### 5. Start the system

```
python start_all.py
```

This starts the orchestrator, the two AI servers and the Streamlit interface.
See DISTRIBUTED_SYSTEM.md for how the parts work together.

## Running single parts

Run everything from the project folder, for example:

```
python pipelines/sms_pipeline.py
python training/sms_model_v1.1.py
python input_router.py
python run_tests.py
```
