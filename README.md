# SE_project_group_8
A school project for the course: Software Engineering for CSAI.


Please check with the person who wrote the code if you remove or change it.

## Project structure

```
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

Run everything from the project folder, for example:

```
python pipelines/sms_pipeline.py
python training/sms_model_v1.1.py
python input_router.py
python run_tests.py
```
