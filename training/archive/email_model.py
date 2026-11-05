#Email model
from pathlib import Path

import joblib
import numpy as np

from scipy.sparse import load_npz

from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# Paths
# ============================================================

base_dir = Path(__file__).resolve().parent.parent

processed_dir = base_dir / "data" / "processed" / "emails"
model_dir = base_dir / "models"

model_dir.mkdir(parents=True, exist_ok=True)


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
# Create Linear SVM
# ============================================================

model = LinearSVC(
    C=1.0,
    random_state=42,
    max_iter=5000
)


# ============================================================
# Train
# ============================================================

print("\nTraining Linear SVM...")

model.fit(X_train, y_train)

print("Training complete.")


# ============================================================
# Validation evaluation
# ============================================================

y_val_pred = model.predict(X_val)

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
        target_names=["ham", "spam"]
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

y_test_pred = model.predict(X_test)

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
        target_names=["ham", "spam"]
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

model_path = model_dir / "email_linear_svm.joblib"

joblib.dump(model, model_path)

print(f"\nModel saved to: {model_path}")

