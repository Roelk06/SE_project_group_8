import numpy as np
import joblib
import torch

from torch import nn
from torchvision import models
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

def validate_image(data):
    if not isinstance(data, (str, Path)):
        raise ValueError("image input must be a file path.")

    image_path = Path(data)

    if not image_path.is_absolute():
        image_path = base_dir / image_path

    if image_path.suffix.lower() not in SUPPORTED_IMAGE_TYPES:
        raise ValueError("Unsupported image format.")

    if not image_path.exists():
        raise ValueError("Image file does not exist.")

    return image_path

def route_input(input_type, data):
    input_type = input_type.lower()

    if input_type not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported input type: {input_type}")

    if input_type == "sms":
        validated_data = validate_sms(data)
        processed_data = preprocess_sms_input(validated_data)
        model = joblib.load(sms_model_path)
        prediction = model.predict(processed_data)[0]
        return "spam" if prediction == 1 else "ham"
    

    if input_type == "email": 
        validated_data = validate_email(data)
        processed_data = preprocess_email_input(validated_data)
        model = joblib.load(email_model_path)
        prediction = model.predict(processed_data)[0]
        return "spam" if prediction == 1 else "ham"

    if input_type == "url":
        validated_data = validate_url(data)
        processed_data = preprocess_url_input(validated_data)
        model = joblib.load(url_model_path)
        prediction = model.predict(processed_data)[0]
        return "phishing" if prediction == 1 else "legitimate"

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
        return "spam" if spam_probability >= spam_threshold else "ham"

    return f"{input_type} routing not implemented yet."

if __name__ == "__main__":

#tests
    print(route_input("sms", "Hey, are you coming to training tonight?"))
    print(route_input("sms", "Can you send me the notes from class?"))
    print(route_input("sms", "Congratulations! You have won £5000. Reply WIN now!"))
    print(route_input("sms", "FREE prize waiting for you! Click now to claim your reward."))
    print(route_input("sms", "I'll be home around 8, do you need anything from the store?"))

    print(route_input("email", {
            "subject": "Meeting tomorrow",
            "message": "Hi, are we still meeting tomorrow at 10?"}))
    print(route_input("email", {
            "subject": "Assignment",
            "message": "I attached the document we discussed during class."}))
    print(route_input("email", {
            "subject": "YOU HAVE WON!",
            "message": "Congratulations! You have been selected for a cash prize. Click here to claim it now."}))
    print(route_input("email", {
            "subject": "URGENT ACCOUNT NOTICE",
            "message": "Your account will be suspended. Verify your information immediately by clicking this link."}))
    print(route_input("email", {
            "subject": "Dinner",
            "message": "Do you want to grab dinner after work tonight?"}))

    print(route_input("url", "https://www.google.com"))
    print(route_input("url", "https://www.tilburguniversity.edu"))
    print(route_input("url", "http://free-prize-login-account.xyz"))
    print(route_input("url", "http://verify-paypal-account-now.ru"))
    print(route_input("url", "https://github.com"))

    print(route_input("image", "data/raw/images/ham/zzz_test_32.jpg"))
    print(route_input("image", "data/raw/images/ham/zzz_test_36.JPG"))
    print(route_input("image", "data/raw/images/ham/zzz_test_45.jpg"))
    print(route_input("image", "data/raw/images/spam/undecided.jpg"))
    print(route_input("image", "data/raw/images/spam/underclassmen.6.jpg"))