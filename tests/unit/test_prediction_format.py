"""
Unit tests for the standardized output: {"class": "spam", "confidence": 0.93}

They test the predict_with_confidence function of each model on tiny toy data,
so they need no real dataset, no saved models and run in a few seconds.
"""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from helpers import (
    EXPECTED_CLASS_NAMES,
    assert_standard_result,
    load_prediction_function,
    make_sparse_data,
)

pytestmark = pytest.mark.unit  # every test in this file is a unit test

ALL_SCRIPTS = ["sms_model.py", "email_model.py", "images_model.py", "url_model.py"]
SVM_SCRIPTS = ["sms_model.py", "email_model.py"]


@pytest.fixture(scope="module")
def toy_svm():
    """A LinearSVC trained on toy data (same settings as the SMS/email scripts)."""
    X, y = make_sparse_data(150, seed=1)
    model = LinearSVC(C=1.0, random_state=42, max_iter=5000).fit(X, y)
    return model, X


class FakeURLModel:
    """
    Stands in for the URL logistic regression so the threshold logic can be tested
    with exact numbers. The "features" X are simply the decision scores.
    """

    def decision_function(self, X):
        return np.asarray(X, dtype=float)

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-np.asarray(X, dtype=float)))
        return np.column_stack([1.0 - p, p])


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-value))


# ------------------------------------------------------------
# UT-1  class names
# ------------------------------------------------------------
@pytest.mark.parametrize("script", ALL_SCRIPTS)
def test_class_names_are_ham_then_spam(script):
    """
    Purpose: every model must name its classes ham (0) and spam (1), in that order.
    Test data: the CLASS_NAMES list read from each model script.
    Expected: CLASS_NAMES == ["ham", "spam"] for sms, email, images and url.
    """
    _, class_names = load_prediction_function(script)
    assert class_names == EXPECTED_CLASS_NAMES


# ------------------------------------------------------------
# UT-2  SVM output format
# ------------------------------------------------------------
@pytest.mark.parametrize("script", SVM_SCRIPTS)
def test_svm_output_has_standard_format(script, toy_svm):
    """
    Purpose: the SMS and email models return one correctly formatted dict per sample.
    Test data: a LinearSVC trained on 150 toy samples, then asked to classify 5 of them.
    Expected: 5 results, each with exactly the keys class and confidence, class is
    ham or spam, confidence is a float between 0.5 and 1.0.
    """
    predict_with_confidence, _ = load_prediction_function(script)
    model, X = toy_svm

    results = predict_with_confidence(model, X[:5])

    assert len(results) == 5
    for result in results:
        assert_standard_result(result)
        assert result["confidence"] >= 0.5  # SVM confidence is a 0.5 - 1.0 score


# ------------------------------------------------------------
# UT-3  SVM class matches the model's own prediction
# ------------------------------------------------------------
@pytest.mark.parametrize("script", SVM_SCRIPTS)
def test_svm_class_matches_model_prediction(script, toy_svm):
    """
    Purpose: the class in the standardized output must be the class the model really predicts.
    Test data: 150 toy samples and the SVM trained on them.
    Expected: result["class"] equals model.predict() (0 = ham, 1 = spam) for every sample.
    """
    predict_with_confidence, _ = load_prediction_function(script)
    model, X = toy_svm

    results = predict_with_confidence(model, X)
    expected = ["spam" if p == 1 else "ham" for p in model.predict(X)]

    assert [r["class"] for r in results] == expected


# ------------------------------------------------------------
# UT-4  SVM confidence grows with distance from the boundary
# ------------------------------------------------------------
@pytest.mark.parametrize("script", SVM_SCRIPTS)
def test_svm_confidence_is_higher_far_from_boundary(script, toy_svm):
    """
    Purpose: a sample far from the decision boundary must get higher confidence than a
    sample right next to it, and the value must follow the documented formula
    confidence = 1 / (1 + e^-|decision score|).
    Test data: the toy sample with the largest |decision score| and the one with the smallest.
    Expected: confidence(far) > confidence(near), and both match the formula.
    """
    predict_with_confidence, _ = load_prediction_function(script)
    model, X = toy_svm

    scores = np.abs(model.decision_function(X))
    far, near = int(np.argmax(scores)), int(np.argmin(scores))

    far_result = predict_with_confidence(model, X[far])[0]
    near_result = predict_with_confidence(model, X[near])[0]

    assert far_result["confidence"] > near_result["confidence"]
    assert far_result["confidence"] == pytest.approx(sigmoid(scores[far]), abs=1e-4)
    assert near_result["confidence"] == pytest.approx(sigmoid(scores[near]), abs=1e-4)


# ------------------------------------------------------------
# UT-5  single sample input
# ------------------------------------------------------------
@pytest.mark.parametrize("script", SVM_SCRIPTS)
def test_svm_accepts_a_single_sample(script, toy_svm):
    """
    Purpose: the program will often classify just one message at a time.
    Test data: exactly one row of toy features.
    Expected: a list with exactly one correctly formatted result.
    """
    predict_with_confidence, _ = load_prediction_function(script)
    model, X = toy_svm

    results = predict_with_confidence(model, X[:1])

    assert len(results) == 1
    assert_standard_result(results[0])


# ------------------------------------------------------------
# UT-6  images: threshold logic
# ------------------------------------------------------------
def test_images_uses_the_tuned_spam_threshold():
    """
    Purpose: the image model decides spam with the tuned threshold, not a fixed 0.5.
    Test data: spam probabilities [0.95, 0.80, 0.79, 0.10] with threshold 0.80.
    Expected: classes [spam, spam, ham, ham] (a value equal to the threshold counts as
    spam); confidence is the probability of the predicted class: [0.95, 0.80, 0.21, 0.90].
    """
    predict_with_confidence, _ = load_prediction_function("images_model.py")

    results = predict_with_confidence(np.array([0.95, 0.80, 0.79, 0.10]), 0.80)

    assert [r["class"] for r in results] == ["spam", "spam", "ham", "ham"]
    expected_confidence = [0.95, 0.80, 0.21, 0.90]
    for result, expected in zip(results, expected_confidence):
        assert_standard_result(result)
        assert result["confidence"] == pytest.approx(expected, abs=1e-4)


# ------------------------------------------------------------
# UT-7  images: known low-confidence ham case
# ------------------------------------------------------------
def test_images_ham_below_threshold_can_have_confidence_under_half():
    """
    Purpose: document a known behaviour. With a high tuned threshold, an image that is
    "probably spam" but below the threshold is called ham with confidence under 0.5.
    Test data: spam probability 0.70 with threshold 0.90.
    Expected: class ham, confidence 0.30 (still a valid number between 0 and 1).
    """
    predict_with_confidence, _ = load_prediction_function("images_model.py")

    result = predict_with_confidence(np.array([0.70]), 0.90)[0]

    assert_standard_result(result)
    assert result["class"] == "ham"
    assert result["confidence"] == pytest.approx(0.30, abs=1e-4)


# ------------------------------------------------------------
# UT-8  url: threshold logic
# ------------------------------------------------------------
def test_url_uses_the_tuned_threshold_on_the_score():
    """
    Purpose: the URL model compares the raw score with its tuned threshold, and reports
    the probability of whichever class it picked.
    Test data: scores [-3, 0.5, 1.0, 4] with threshold 1.0, using a stand-in model.
    Expected: classes [ham, ham, spam, spam]; confidence = 1 - sigmoid(score) for ham
    and sigmoid(score) for spam.
    """
    predict_with_confidence, _ = load_prediction_function("url_model.py")
    scores = np.array([-3.0, 0.5, 1.0, 4.0])

    results = predict_with_confidence(FakeURLModel(), scores, 1.0)

    assert [r["class"] for r in results] == ["ham", "ham", "spam", "spam"]
    expected = [1 - sigmoid(-3.0), 1 - sigmoid(0.5), sigmoid(1.0), sigmoid(4.0)]
    for result, value in zip(results, expected):
        assert_standard_result(result)
        assert result["confidence"] == pytest.approx(value, abs=1e-4)


# ------------------------------------------------------------
# UT-9  all four models share one schema
# ------------------------------------------------------------
def test_all_four_models_return_the_same_schema(toy_svm):
    """
    Purpose: the program that reads the results must be able to treat all models the same.
    Test data: one output from each of the sms, email, images and url functions.
    Expected: every result has the keys class and confidence (in that order), a class of
    ham/spam and a float confidence between 0 and 1.
    """
    model, X = toy_svm
    outputs = []

    for script in SVM_SCRIPTS:
        function, _ = load_prediction_function(script)
        outputs += function(model, X[:2])

    function, _ = load_prediction_function("images_model.py")
    outputs += function(np.array([0.9, 0.1]), 0.5)

    function, _ = load_prediction_function("url_model.py")
    outputs += function(FakeURLModel(), np.array([2.0, -2.0]), 0.0)

    assert len(outputs) == 8
    for result in outputs:
        assert_standard_result(result)
        assert list(result.keys()) == ["class", "confidence"]


# ------------------------------------------------------------
# UT-10  rounding
# ------------------------------------------------------------
def test_confidence_is_rounded_to_four_decimals():
    """
    Purpose: confidence values are rounded so the output is clean for the program/user.
    Test data: spam probability 0.123456789 with threshold 0.5 (image function).
    Expected: ham with confidence exactly 0.8765.
    """
    predict_with_confidence, _ = load_prediction_function("images_model.py")

    result = predict_with_confidence(np.array([0.123456789]), 0.5)[0]

    assert result["class"] == "ham"
    assert result["confidence"] == 0.8765
