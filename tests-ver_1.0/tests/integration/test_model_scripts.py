"""
Integration tests: run each model script from start to finish (load data -> train ->
evaluate -> save model) in a throw-away copy of the project folder, then load the saved
model again and use it.

This checks that the pieces work together: the data folder layout, the training code,
the saved model file and the standardized prediction function.
"""
import joblib
import pytest

from helpers import (
    assert_standard_result,
    outdated_hint,
    load_prediction_function,
    make_sparse_data,
    run_script_in_sandbox,
    write_image_dataset,
    write_sparse_dataset,
)

pytestmark = pytest.mark.integration  # every test in this file is an integration test

# text every script prints when it ran correctly
REQUIRED_OUTPUT = [
    "TRAINING STATISTICS",
    "Training time",
    "VALIDATION RESULTS",
    "TEST RESULTS",
    "Inference time (whole test set)",
    "EXAMPLE PREDICTIONS",
]

SVM_CASES = [
    ("sms_model.py", "SMS", "sms_linear_svm.joblib"),
    ("email_model.py", "emails", "email_linear_svm.joblib"),
]


def assert_script_succeeded(result, script):
    hint = outdated_hint(script)  # empty unless the script is still the original version
    assert result.returncode == 0, (
        f"{hint}script crashed (exit code {result.returncode})\n"
        f"--- stderr (last part) ---\n{result.stderr[-1500:]}"
    )
    for marker in REQUIRED_OUTPUT:
        assert marker in result.stdout, f"{hint}missing in output: {marker!r}"


# ------------------------------------------------------------
# IT-1  SMS and email: data -> train -> save -> load -> predict
# ------------------------------------------------------------
@pytest.mark.parametrize("script, data_folder, model_file", SVM_CASES, ids=["sms", "email"])
def test_svm_script_runs_end_to_end(script, data_folder, model_file, tmp_path):
    """
    Purpose: the SMS and email scripts train, evaluate and save a working model.
    Test data: toy sparse data (150 train / 50 val / 50 test) written into
    data/processed/<SMS or emails>/ of a temporary project; script run from models/.
    Expected: exit code 0, all report sections printed, the .joblib model is saved,
    and the reloaded model gives correctly formatted predictions on new samples.
    """
    result, project = run_script_in_sandbox(script, data_folder, write_sparse_dataset, tmp_path)
    assert_script_succeeded(result, script)

    model_path = project / "models" / model_file
    assert model_path.exists(), "the trained model was not saved"

    model = joblib.load(model_path)
    predict_with_confidence, _ = load_prediction_function(script)
    new_samples, _ = make_sparse_data(3, seed=99)

    results = predict_with_confidence(model, new_samples)
    assert len(results) == 3
    for item in results:
        assert_standard_result(item)


# ------------------------------------------------------------
# IT-2  URL: data -> train -> tune threshold -> save -> load -> predict
# ------------------------------------------------------------
def test_url_script_runs_end_to_end(tmp_path):
    """
    Purpose: the URL script trains, tunes its threshold, saves the model together with
    that threshold, and the saved result can be used for standardized predictions.
    Test data: toy sparse data (150 train / 50 val / 50 test) in data/processed/urls/
    of a temporary project; script run from inside models/.
    Expected: exit code 0, all report sections printed, url_logistic_regression.joblib
    saved with a bad_threshold_ value, and correctly formatted predictions using it.
    """
    result, project = run_script_in_sandbox(
        "url_model.py", "urls", write_sparse_dataset, tmp_path
    )
    assert_script_succeeded(result, "url_model.py")
    assert "Selected bad threshold" in result.stdout

    model_path = project / "models" / "url_logistic_regression.joblib"
    assert model_path.exists(), "the trained model was not saved"

    model = joblib.load(model_path)
    assert hasattr(model, "bad_threshold_"), "the tuned threshold was not saved with the model"

    predict_with_confidence, _ = load_prediction_function("url_model.py")
    new_samples, _ = make_sparse_data(3, seed=99)

    results = predict_with_confidence(model, new_samples, model.bad_threshold_)
    assert len(results) == 3
    for item in results:
        assert_standard_result(item)


# ------------------------------------------------------------
# IT-3  Images: data -> train (ResNet-18) -> save checkpoint
# ------------------------------------------------------------
def test_images_script_runs_end_to_end(tmp_path):
    """
    Purpose: the image script trains, evaluates and saves a checkpoint with a spam threshold.
    Test data: toy images shaped (samples, 3, 32, 32) (64 train / 32 val / 32 test) in
    data/processed/images/ of a temporary project. To keep the test fast and offline the
    COPY of the script uses 1 epoch and no pretrained weights (the real script is unchanged).
    Expected: exit code 0, all report sections printed, image_resnet18.pt saved with the
    class names and a spam threshold between 0.5 and 1.0.
    This test is skipped automatically when torch / torchvision are not installed.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    result, project = run_script_in_sandbox(
        "images_model.py",
        "images",
        write_image_dataset,
        tmp_path,
        text_edits=[
            ("weights = models.ResNet18_Weights.DEFAULT", "weights = None"),
            ("epochs = 8", "epochs = 1"),
        ],
    )
    assert_script_succeeded(result, "images_model.py")
    assert "Epochs:" in result.stdout

    checkpoint_path = project / "models" / "image_resnet18.pt"
    assert checkpoint_path.exists(), "the checkpoint was not saved"

    checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert checkpoint["class_names"] == ["ham", "spam"]
    assert 0.5 <= checkpoint["spam_threshold"] <= 1.0
