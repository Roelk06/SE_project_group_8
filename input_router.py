SUPPORTED_TYPES = {"sms", "email", "url", "image"}

def validate_sms(data):
    if not isinstance(data, str) or not data.strip():
        raise ValueError("SMS input must have text.")
    return data

def route_input(input_type, data):
    input_type = input_type.lower()
    if input_type not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported input type: {input_type}")

    if input_type == "sms":
        validate_sms(data)
        return "SMS input routed succesfully!"

    return f"{input_type} routing not implemented yet."

#tests
print(route_input("sms", "Hello this is a test message"))
#print(route_input("sms", "   "))
#print(route_input("video", "test"))
