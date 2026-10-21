#SMS model
import time
from pathlib import Path
 
import joblib
import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.svm import LinearSVC
 
 
# ============================================================
# Paths
# ============================================================
 
base_dir = Path(__file__).resolve().parent.parent
 
processed_dir = base_dir / "data" / "processed" / "SMS"
model_dir = base_dir / "models"
 
model_dir.mkdir(parents=True, exist_ok=True)
 
 
# ============================================================
# Standardized prediction output
# ============================================================
 
CLASS_NAMES = ["ham", "spam"]
 
 
def predict_with_confidence(model, X):
    """
    Returns a list of dicts like {"class": "spam", "confidence": 0.93}
    (one per row in X).
 
    LinearSVC has no predict_proba, so the confidence is made from the
    decision score: the further a sample is from the boundary, the closer
    the confidence gets to 1.0. It is not a true probability, just a
    0.5 - 1.0 score for the predicted class.
    """
    scores = model.decision_function(X)
    preds = model.predict(X)
 
    results = []
    for pred, score in zip(preds, scores):
        confidence = 1.0 / (1.0 + np.exp(-abs(score)))
        results.append({
            "class": CLASS_NAMES[int(pred)],
            "confidence": round(float(confidence), 4),
        })
    return results
 
 
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
# Create model
# ============================================================
 
model = LinearSVC(
    C=1.0,
    random_state=42,
    max_iter=5000
)
 
 
# ============================================================
# Train model
# ============================================================
 
print("\nTraining Linear SVM...")
 
start_time = time.perf_counter()
model.fit(X_train, y_train)
training_time = time.perf_counter() - start_time
 
print("Training complete.")
 
 
# ============================================================
# Training statistics
# ============================================================
 
n_iterations = int(np.max(model.n_iter_))
 
print("\n" + "=" * 60)
print("TRAINING STATISTICS")
print("=" * 60)
 
print(f"Training time:      {training_time:.4f} seconds")
print(f"Iterations used:    {n_iterations} (max allowed: {model.max_iter})")
 
 
# ============================================================
# Evaluate on validation set
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
print(confusion_matrix(y_val, y_val_pred))
 
 
# ============================================================
# Evaluate on test set
# ============================================================
 
start_time = time.perf_counter()
y_test_pred = model.predict(X_test)
inference_time = time.perf_counter() - start_time
 
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
print(confusion_matrix(y_test, y_test_pred))
 
print(f"\nInference time (whole test set): {inference_time:.4f} seconds")
print(f"Inference time (per sample):     {inference_time / X_test.shape[0] * 1000:.4f} ms")
 
 
# ============================================================
# Example of standardized output
# ============================================================
 
print("\n" + "=" * 60)
print("EXAMPLE PREDICTIONS (standardized output)")
print("=" * 60)
 
example_results = predict_with_confidence(model, X_test[:5])
 
for i, result in enumerate(example_results):
    actual = CLASS_NAMES[int(y_test[i])]
    print(f"Sample {i}: {result}  (actual: {actual})")
 
 
# ============================================================
# Save trained model
# ============================================================
 
model_path = model_dir / "sms_linear_svm.joblib"
 
joblib.dump(model, model_path)
 
print(f"\nModel saved to: {model_path}")