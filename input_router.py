from urllib.parse import urlparse
SUPPORTED_TYPES = {"sms", "email", "url", "image"}

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

def route_input(input_type, data):
    input_type = input_type.lower()

    if input_type not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported input type: {input_type}")

    if input_type == "sms":
        validate_sms(data)
        return "SMS input routed succesfully!"

    if input_type == "email": 
        validate_email(data)
        return "Email input routed succesfully!"

    if input_type == "url":
        validate_url(data)
        return "URL input routed succesfully!"

    return f"{input_type} routing not implemented yet."

print(route_input("url", "ftp://example.com"))