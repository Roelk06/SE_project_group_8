import json
import shutil
import uuid

import numpy as np
import joblib
import torch

from torch import nn
from torchvision import models
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from pipelines.sms_pipeline import preprocess_sms_input
from pipelines.email_pipeline import preprocess_email_input
from pipelines.url_pipeline import preprocess_url_input
from pipelines.images_pipeline import preprocess_image_input

SUPPORTED_TYPES = {"sms", "email", "url", "image"}
SUPPORTED_IMAGE_TYPES = {".jpg", ".jpeg"}

base_dir = Path(__file__).resolve().parent

sms_model_path = base_dir / "models" / "sms_linear_svm.joblib"
email_model_path = base_dir / "models" / "email_linear_svm.joblib"
url_model_path = base_dir / "models" / "url_logistic_regression.joblib"
image_model_path = base_dir / "models" / "image_resnet18.pt"

RESULTS_DIR = base_dir / "results"
LABEL_TO_FOLDER = {
    "spam": "spam",
    "phishing": "spam",
    "ham": "ham",
    "legitimate": "ham",
}

def validate_sms(data):
    if not isinstance(data, str) or not data.strip():
        raise ValueError("SMS input must have text.")
    
    return data

def validate_email(data):
    if not isinstance(data, dict) or "subject" not in data or "message" not in data:
        raise ValueError("Email input must have a subject and message.")

    if not isinstance(data["subject"], str) or not isinstance(data["message"], str):
        raise ValueError("Email subject and message must be text.")

    if not data["subject"].strip() and not data["message"].strip():
        raise ValueError("Email must have text.")

    return data

def validate_url(data):
    if not isinstance(data, str) or not data.strip():
        raise ValueError("URL input must have text.")

    url = data.strip()
    parsed = urlparse(url if "://" in url else "//" + url)

    if not parsed.netloc or "." not in parsed.netloc:
        raise ValueError("Invalid URL.")

    if parsed.scheme and parsed.scheme not in ("http", "https"):
        raise ValueError("Unsupported URL scheme.")

    return data

def resolve_image_path(data):
    image_path = Path(data)

    if not image_path.is_absolute():
        image_path = base_dir / image_path

    return image_path

def validate_image(data):
    if not isinstance(data, (str, Path)):
        raise ValueError("image input must be a file path.")

    image_path = resolve_image_path(data)

    if image_path.suffix.lower() not in SUPPORTED_IMAGE_TYPES:
        raise ValueError("Unsupported image format.")

    if not image_path.exists():
        raise ValueError("Image file does not exist.")

    return image_path

def svm_prediction(model, processed_data):
    # same confidence as predict_with_confidence in the SMS and email training scripts:
    # the further from the decision boundary, the closer to 1.0
    score = model.decision_function(processed_data)[0]
    prediction = model.predict(processed_data)[0]
    confidence = 1.0 / (1.0 + np.exp(-abs(score)))
    return ("spam" if prediction == 1 else "ham"), round(float(confidence), 4)

def route_input(input_type, data):
    # returns (label, confidence), for example ("spam", 0.93)
    input_type = input_type.lower()

    if input_type not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported input type: {input_type}")

    if input_type == "sms":
        validated_data = validate_sms(data)
        processed_data = preprocess_sms_input(validated_data)
        model = joblib.load(sms_model_path)
        return svm_prediction(model, processed_data)
    

    if input_type == "email": 
        validated_data = validate_email(data)
        processed_data = preprocess_email_input(validated_data)
        model = joblib.load(email_model_path)
        return svm_prediction(model, processed_data)

    if input_type == "url":
        validated_data = validate_url(data)
        processed_data = preprocess_url_input(validated_data)
        model = joblib.load(url_model_path)
        # use the threshold tuned in training (saved inside the model), not the default cutoff;
        # same as predict_with_confidence in the URL training script
        score = model.decision_function(processed_data)[0]
        bad_probability = model.predict_proba(processed_data)[0, 1]
        if score >= model.bad_threshold_:
            return "phishing", round(float(bad_probability), 4)
        return "legitimate", round(float(1.0 - bad_probability), 4)

    if input_type == "image":
        validated_data = validate_image(data)
        processed_data = preprocess_image_input(validated_data)
        processed_data = np.transpose(processed_data, (2,0,1))
        processed_data = np.expand_dims(processed_data, axis=0)
        image_tensor = torch.from_numpy(processed_data).float()

        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        checkpoint = torch.load(image_model_path, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        spam_threshold = checkpoint["spam_threshold"]
        model.eval()

        with torch.no_grad():
            output = model(image_tensor)
            spam_probability = torch.softmax(output, dim=1)[0,1].item()
        # same as predict_with_confidence in the image training script
        if spam_probability >= spam_threshold:
            return "spam", round(spam_probability, 4)
        return "ham", round(1.0 - spam_probability, 4)

    raise ValueError(f"{input_type} routing not implemented yet.")

def save_result(input_type, data, label, results_dir=RESULTS_DIR, confidence=None):
    # an unknown label raises a KeyError here, so it can never be let through as not spam
    folder_name = LABEL_TO_FOLDER[label]

    folder = Path(results_dir) / folder_name
    folder.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    # the random part keeps two results from the same moment from overwriting each other
    file_stem = f"{now:%Y%m%d-%H%M%S}_{input_type}_{uuid.uuid4().hex[:8]}"

    content = data
    stored_file = None
    if input_type == "image":
        image_path = resolve_image_path(data)
        stored_path = folder / f"{file_stem}{image_path.suffix.lower()}"
        shutil.copy2(image_path, stored_path)  # the original image stays where it is
        content = str(image_path)
        stored_file = stored_path.name

    record = {
        "timestamp": now.isoformat(timespec="seconds"),
        "input_type": input_type,
        "label": folder_name,
        "model_label": label,
        "confidence": confidence,
        "content": content,
        "stored_file": stored_file,
    }
    record_path = folder / f"{file_stem}.json"
    record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "label": folder_name,
        "allowed": folder_name == "ham",
        "confidence": confidence,
        "record_path": record_path,
    }

def route_and_sort(input_type, data, results_dir=RESULTS_DIR):
    # spam/phishing goes to results/spam (blocked), ham/legitimate to results/ham (let through)
    label, confidence = route_input(input_type, data)
    return save_result(input_type.lower(), data, label, results_dir, confidence)
