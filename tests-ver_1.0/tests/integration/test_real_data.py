"""
Integration tests on the REAL processed data and the REAL saved models.

These only run when the files exist (data/processed/... and the saved models). If a
folder is missing the test is skipped, not failed, so the suite also works on a
machine that does not have the dataset.
"""
import joblib
import numpy as np
import pytest
from scipy.sparse import load_npz

from helpers import ROOT, assert_standard_result, load_prediction_function

pytestmark = pytest.mark.integration  # every test in this file is an integration test

DATASETS = [
    # (id, folder under data/processed, kind of x file)
    ("sms", "SMS", "sparse"),
    ("email", "emails", "sparse"),
    ("url", "urls", "sparse"),
    ("images", "images", "dense"),
]

MODELS = [
    # (id, data folder, script, saved model file)
    ("sms", "SMS", "sms_model.py", "sms_linear_svm.joblib"),
    ("email", "emails", "email_model.py", "email_linear_svm.joblib"),
    ("url", "urls", "url_model.py", "url_logistic_regression.joblib"),
]


def load_split(data_dir, kind, split):
    if kind == "sparse":
        x = load_npz(data_dir / f"x_{split}.npz")
    else:
        x = np.load(data_dir / f"x_{split}.npy", mmap_mode="r")
    y = np.load(data_dir / f"y_{split}.npy")
    return x, y


# ------------------------------------------------------------
# IT-4  processed data is consistent
# ------------------------------------------------------------
@pytest.mark.parametrize(
    "folder, kind", [(d[1], d[2]) for d in DATASETS], ids=[d[0] for d in DATASETS]
)
def test_real_processed_data_is_consistent(folder, kind):
    """
    Purpose: the output of the preprocessing pipelines can be read by the models.
    Test data: the real files in data/processed/<type>/ (x_train, x_val, x_test and
    y_train, y_val, y_test).
    Expected: for each split the number of samples matches the number of labels, labels
    are only 0 and 1 with both present, and all splits have the same feature shape.
    Skipped when the data folder does not exist.
    """
    data_dir = ROOT / "data" / "processed" / folder
    if not data_dir.exists():
        pytest.skip(f"{data_dir} not found")

    feature_shapes = set()
    for split in ["train", "val", "test"]:
        x, y = load_split(data_dir, kind, split)

        assert x.shape[0] == len(y), f"{split}: {x.shape[0]} samples but {len(y)} labels"
        assert set(np.unique(y)) == {0, 1}, f"{split}: labels must be 0 and 1, got {np.unique(y)}"
        feature_shapes.add(tuple(x.shape[1:]))

    assert len(feature_shapes) == 1, f"splits have different feature shapes: {feature_shapes}"


# ------------------------------------------------------------
# IT-5  saved models work on the real data
# ------------------------------------------------------------
@pytest.mark.parametrize(
    "folder, script, model_file",
    [(m[1], m[2], m[3]) for m in MODELS],
    ids=[m[0] for m in MODELS],
)
def test_saved_model_predicts_on_real_test_data(folder, script, model_file):
    """
    Purpose: the model saved by the training script can be loaded later and used on the
    real processed test data, and it returns the standardized output.
    Test data: the first 5 rows of the real x_test, and the saved model in models/.
    Expected: 5 correctly formatted results (class ham/spam, confidence 0 to 1).
    Skipped when the saved model or the data does not exist.
    """
    model_path = ROOT / "models" / model_file
    x_path = ROOT / "data" / "processed" / folder / "x_test.npz"
    if not model_path.exists():
        pytest.skip(f"{model_path} not found (train the model first)")
    if not x_path.exists():
        pytest.skip(f"{x_path} not found")

    model = joblib.load(model_path)
    X = load_npz(x_path)[:5]
    predict_with_confidence, _ = load_prediction_function(script)

    if script == "url_model.py":
        results = predict_with_confidence(model, X, model.bad_threshold_)
    else:
        results = predict_with_confidence(model, X)

    assert len(results) == 5
    for result in results:
        assert_standard_result(result)
