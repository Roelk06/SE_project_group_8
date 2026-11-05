#Images model
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models


# ============================================================
# Paths and settings
# ============================================================


base_dir = Path(__file__).resolve().parent
processed_dir = base_dir / "data" / "processed" / "images"
model_dir = base_dir / "models"
model_dir.mkdir(parents=True, exist_ok=True)

batch_size = 32
epochs = 8
learning_rate = 1e-3
minimum_spam_recall = 0.20

class ImageArrayDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.from_numpy(images).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


# ============================================================
# Load processed data
# ============================================================

train_images = np.load(processed_dir / "x_train.npy")
val_images = np.load(processed_dir / "x_val.npy")
test_images = np.load(processed_dir / "x_test.npy")
train_labels = np.load(processed_dir / "y_train.npy")
val_labels = np.load(processed_dir / "y_val.npy")
test_labels = np.load(processed_dir / "y_test.npy")

if train_images.ndim != 4 or train_images.shape[1] != 3:
    raise ValueError("Expected image arrays with shape (samples, 3, height, width).")

print("Data loaded successfully.")
print(f"Training samples:   {train_images.shape[0]}")
print(f"Validation samples: {val_images.shape[0]}")
print(f"Test samples:       {test_images.shape[0]}")
print(f"Image shape:        {train_images.shape[1:]}")

train_dataset = ImageArrayDataset(train_images, train_labels)
val_dataset = ImageArrayDataset(val_images, val_labels)
test_dataset = ImageArrayDataset(test_images, test_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
# ============================================================
# Create pretrained ResNet-18
# ============================================================

test_loader = DataLoader(test_dataset, batch_size=batch_size)

weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
for parameter in model.parameters():
    parameter.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 2)

class_counts = np.bincount(train_labels, minlength=2)
class_weights = len(train_labels) / (2 * class_counts)
loss_function = nn.CrossEntropyLoss(
    weight=torch.tensor(class_weights, dtype=torch.float32)
)
optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)


def predict(data_loader):
    model.eval()
    spam_probabilities = []
    actual = []
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            spam_probabilities.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
            actual.extend(labels.numpy())
    return np.array(actual), np.array(spam_probabilities)


def tune_spam_threshold(actual, spam_probabilities):
    best_threshold = 0.5
    best_precision = -1.0
    best_recall = 0.0

    for threshold in np.arange(0.50, 1.00, 0.01):
        predictions = (spam_probabilities >= threshold).astype(int)
        precision = precision_score(actual, predictions, zero_division=0)
        recall = recall_score(actual, predictions, zero_division=0)

        if recall >= minimum_spam_recall and (
            precision > best_precision
            or (precision == best_precision and recall > best_recall)
        ):
            best_threshold = threshold
            best_precision = precision
            best_recall = recall

    return best_threshold, best_precision, best_recall


def print_metrics(name, actual, predictions):
    print(f"\n{name} RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {accuracy_score(actual, predictions):.4f}")
    print(f"Precision: {precision_score(actual, predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(actual, predictions, zero_division=0):.4f}")
    print(f"F1 score:  {f1_score(actual, predictions, zero_division=0):.4f}")
    print(classification_report(actual, predictions, target_names=["ham", "spam"], zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(actual, predictions))


# ============================================================
# Train
# ============================================================

best_precision = -1.0
best_recall = -1.0
checkpoint_path = model_dir / "image_resnet18.pt"
print("\nTraining pretrained ResNet-18...")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(labels)

    val_actual, val_probabilities = predict(val_loader)
    threshold, val_precision, val_recall = tune_spam_threshold(
        val_actual, val_probabilities
    )
    average_loss = running_loss / len(train_dataset)
    print(
        f"Epoch {epoch + 1}/{epochs} - loss: {average_loss:.4f} - "
        f"val precision: {val_precision:.4f} - val recall: {val_recall:.4f} "
        f"- threshold: {threshold:.2f}"
    )

    if val_precision > best_precision or (
        val_precision == best_precision and val_recall > best_recall
    ):
        best_precision = val_precision
        best_recall = val_recall
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": ["ham", "spam"],
                "spam_threshold": float(threshold),
            },
            checkpoint_path,
        )


# ============================================================
# Validation evaluation
# ============================================================

checkpoint = torch.load(checkpoint_path, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
spam_threshold = checkpoint["spam_threshold"]
val_actual, val_probabilities = predict(val_loader)
val_predictions = (val_probabilities >= spam_threshold).astype(int)
print(f"\nUsing spam threshold: {spam_threshold:.2f}")
print_metrics("VALIDATION", val_actual, val_predictions)


# ============================================================
# Test evaluation
# ============================================================

test_actual, test_probabilities = predict(test_loader)
test_predictions = (test_probabilities >= spam_threshold).astype(int)
print_metrics("TEST", test_actual, test_predictions)


# ============================================================
# Save model
# ============================================================

print(f"Best model saved to: {checkpoint_path}")

