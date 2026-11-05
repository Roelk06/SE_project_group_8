#URL model
from pathlib import Path

import joblib
import numpy as np

from scipy.sparse import load_npz

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve
)


# ============================================================
# Paths
# ============================================================

base_dir = Path(__file__).resolve().parent

processed_dir = base_dir / "data" / "processed" / "urls"
model_dir = base_dir / "models"

model_dir.mkdir(parents=True, exist_ok=True)

minimum_bad_precision = 0.80


# ============================================================
# Load processed data
# ============================================================

X_train = load_npz(processed_dir / "x_train.npz")
X_val = load_npz(processed_dir / "x_val.npz")
X_test = load_npz(processed_dir / "x_test.npz")

y_train = np.load(processed_dir / "y_train.npy")
y_val = np.load(processed_dir / "y_val.npy")
y_test = np.load(processed_dir / "y_test.npy")


print("Data loaded successfully.")

print(f"Training samples:   {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")
print(f"Test samples:       {X_test.shape[0]}")
print(f"Number of features: {X_train.shape[1]}")


# ============================================================
# Create Logistic Regression model
# ============================================================

model = LogisticRegression(
    C=4.0,
    max_iter=100,
    solver="saga"
)


# ============================================================
# Train
# ============================================================

print("\nTraining Logistic Regression...")

model.fit(X_train, y_train)

print("Training complete.")


# ============================================================
# Tune decision threshold for precision
# ============================================================

val_scores = model.decision_function(X_val)
precision_values, recall_values, thresholds = precision_recall_curve(
    y_val,
    val_scores
)

eligible_thresholds = precision_values[:-1] >= minimum_bad_precision

if not np.any(eligible_thresholds):
    bad_threshold = 0.0
else:
    best_index = np.argmax(
        np.where(eligible_thresholds, recall_values[:-1], -1.0)
    )
    bad_threshold = thresholds[best_index]

print(f"Selected bad threshold: {bad_threshold:.4f}")


# ============================================================
# Validation evaluation
# ============================================================

y_val_pred = (val_scores >= bad_threshold).astype(int)

print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

print(f"Accuracy:  {accuracy_score(y_val, y_val_pred):.4f}")
print(f"Precision: {precision_score(y_val, y_val_pred):.4f}")
print(f"Recall:    {recall_score(y_val, y_val_pred):.4f}")
print(f"F1 score:  {f1_score(y_val, y_val_pred):.4f}")

print("\nClassification report:")

print(
    classification_report(
        y_val,
        y_val_pred,
        target_names=["good", "bad"]
    )
)

print("Confusion matrix:")

print(
    confusion_matrix(
        y_val,
        y_val_pred
    )
)


# ============================================================
# Test evaluation
# ============================================================

test_scores = model.decision_function(X_test)
y_test_pred = (test_scores >= bad_threshold).astype(int)

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(f"Accuracy:  {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_test_pred):.4f}")
print(f"F1 score:  {f1_score(y_test, y_test_pred):.4f}")

print("\nClassification report:")

print(
    classification_report(
        y_test,
        y_test_pred,
        target_names=["good", "bad"]
    )
)

print("Confusion matrix:")

print(
    confusion_matrix(
        y_test,
        y_test_pred
    )
)


# ============================================================
# Save model
# ============================================================

model_path = model_dir / "url_logistic_regression.joblib"

model.bad_threshold_ = float(bad_threshold)
joblib.dump(model, model_path)

print(f"\nModel saved to: {model_path}")

