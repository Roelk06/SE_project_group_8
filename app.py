import base64
import io
import os
import time
from email import policy
from email.parser import BytesParser
from pathlib import Path

import requests
import streamlit as st
from PIL import Image


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Spam Detector - Group 8",
    page_icon="🛡️",
    layout="wide"
)

ORCHESTRATOR_URL = os.getenv(
    "ORCHESTRATOR_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg"}  # same as the router
MAX_IMAGE_MB = 5

if "history" not in st.session_state:
    st.session_state.history = []


# --------------------------------------------------
# COMMUNICATION WITH ORCHESTRATOR
# --------------------------------------------------

def analyze_input(
    modality: str,
    payload,
    source_name: str = "Direct Input"
) -> dict:

    """
    Sends user input to the orchestrator.

    The interface does NOT communicate directly
    with any AI server.
    """

    start_time = time.perf_counter()

    response = requests.post(
        f"{ORCHESTRATOR_URL}/analyze",
        json={
            "modality": modality.lower(),
            "payload": payload,
            "source_name": source_name
        },
        timeout=65
    )

    response.raise_for_status()

    result = response.json()

    result["latency_ms"] = round(
        (time.perf_counter() - start_time) * 1000,
        1
    )

    return result


# --------------------------------------------------
# RESULT DISPLAY
# --------------------------------------------------

def display_result(result: dict, threshold: float):

    st.divider()
    st.subheader("Analysis Result")

    if result.get("status") != "success":
        st.warning("The analysis was not completed.")
        return

    if result["is_spam"]:
        st.error(
            f"⚠️ **Classification: {result['prediction']}**"
        )
    else:
        st.success(
            f"✅ **Classification: {result['prediction']}**"
        )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Detected Modality",
        result["modality"].upper()
    )

    col2.metric(
        "Confidence",
        f"{result['confidence'] * 100:.1f}%"
    )

    col3.metric(
        "Latency",
        f"{result['latency_ms']} ms"
    )

    st.caption(
        f"Processed by: {result.get('ai_server', 'Unknown')} "
        f"| Model: {result.get('model', 'Unknown')}"
    )

    if result.get("stored_in"):
        action = "blocked" if result["is_spam"] else "let through"
        st.caption(f"Stored in: {result['stored_in']} ({action})")

    if result["confidence"] < threshold:
        st.warning(
            "The prediction is below your selected "
            "confidence threshold. Consider reviewing it."
        )


def process_request(modality, payload, source_name, threshold):

    try:
        with st.spinner("Sending request through orchestrator..."):

            result = analyze_input(
                modality,
                payload,
                source_name
            )

            st.session_state.history.insert(0, result)

            display_result(result, threshold)

    except requests.exceptions.ConnectionError:

        st.error(
            "Cannot connect to the orchestrator. "
            "Please check whether all servers are running."
        )

    except requests.exceptions.Timeout:

        st.error(
            "The analysis took too long. Please try again."
        )

    except requests.exceptions.HTTPError as error:

        try:
            detail = error.response.json().get(
                "detail",
                "The server returned an error."
            )
        except Exception:
            detail = str(error)

        st.error(f"Analysis failed: {detail}")

    except Exception as error:

        st.error(f"Unexpected error: {error}")


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("⚙️ Prototype Controls")

    st.caption("SE Project Group 8")

    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.5,
        max_value=0.99,
        value=0.75,
        step=0.05
    )

    st.divider()

    if st.button(
        "🗑️ Clear Scan History",
        use_container_width=True
    ):
        st.session_state.history = []
        st.rerun()

    st.divider()

    st.caption("Backend connection")

    try:
        health = requests.get(
            f"{ORCHESTRATOR_URL}/health",
            timeout=2
        )

        if health.status_code == 200:
            st.success("Orchestrator connected")

            try:
                servers = requests.get(
                    f"{ORCHESTRATOR_URL}/services",
                    timeout=2
                ).json()["servers"]

                if not servers:
                    st.warning("No AI servers registered")

                for name, info in servers.items():
                    marker = "🟢" if info.get("online") else "🔴"
                    st.write(f"{marker} **{name}**: {', '.join(info['types'])}")

            except (requests.exceptions.RequestException, ValueError, KeyError):
                pass
        else:
            st.warning("Orchestrator unavailable")

    except requests.exceptions.RequestException:
        st.error("Orchestrator disconnected")


# --------------------------------------------------
# MAIN INTERFACE
# --------------------------------------------------

st.title("🛡️ Multi-Modal Spam Detector")

st.write(
    "Submit text, images or URLs to check for spam content."
)

tab_text, tab_url, tab_file = st.tabs(
    [
        "📝 Text / Email Body",
        "🔗 URL Check",
        "📂 File / Image Upload"
    ]
)


# --------------------------------------------------
# TEXT ANALYSIS
# --------------------------------------------------

with tab_text:

    user_text = st.text_area(
        "Paste SMS or Email content here:",
        height=150,
        placeholder=(
            "e.g., URGENT: Your account password "
            "expires today. Click here to verify..."
        )
    )

    text_type = st.radio(
        "Input type:",
        ["SMS", "Email"],
        horizontal=True
    )

    email_subject = ""

    if text_type == "Email":
        email_subject = st.text_input("Email subject (optional):")

    if st.button(
        "Analyze Text",
        type="primary",
        key="btn_text"
    ):

        if not user_text.strip():

            st.warning(
                "Please enter some text before analyzing."
            )

        else:

            modality = text_type.lower()

            if modality == "email":
                text_payload = {
                    "subject": email_subject,
                    "message": user_text
                }
            else:
                text_payload = user_text

            process_request(
                modality,
                text_payload,
                "Pasted Text",
                confidence_threshold
            )


# --------------------------------------------------
# URL ANALYSIS
# --------------------------------------------------

with tab_url:

    user_url = st.text_input(
        "Enter a URL to inspect:",
        placeholder="https://example.com"
    )

    if st.button(
        "Analyze URL",
        type="primary",
        key="btn_url"
    ):

        if not user_url.strip():

            st.warning(
                "Please enter a URL before analyzing."
            )

        else:

            process_request(
                "url",
                user_url,
                "Entered URL",
                confidence_threshold
            )


# --------------------------------------------------
# FILE / IMAGE ANALYSIS
# --------------------------------------------------

with tab_file:

    uploaded_file = st.file_uploader(
        "Upload an email (.eml or .txt) or image (.png, .jpg, .jpeg)",
        type=["eml", "txt", "png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:

        file_type = uploaded_file.type
        file_extension = Path(uploaded_file.name).suffix.lower()
        file_bytes = uploaded_file.getvalue()

        if file_type.startswith("image/"):

            try:

                image = Image.open(io.BytesIO(file_bytes))

                st.image(
                    image,
                    caption=f"Uploaded: {uploaded_file.name}",
                    width=350
                )

            except Exception:
                st.error("Could not read the uploaded image.")

            if file_extension not in SUPPORTED_IMAGE_EXTENSIONS:

                st.error(
                    f"Unsupported image format ({file_extension}). "
                    "Please upload a JPG or JPEG image."
                )

            elif len(file_bytes) > MAX_IMAGE_MB * 1024 * 1024:

                st.error(f"This image is larger than {MAX_IMAGE_MB} MB.")

            elif st.button(
                "Analyze Uploaded File",
                type="primary",
                key="btn_file"
            ):

                # The image travels inside the request,
                # because the AI server is a separate program.
                process_request(
                    "image",
                    {
                        "filename": uploaded_file.name,
                        "content_b64": base64.b64encode(file_bytes).decode("ascii")
                    },
                    uploaded_file.name,
                    confidence_threshold
                )

        else:

            if file_extension == ".eml":

                email_message = BytesParser(policy=policy.default).parsebytes(file_bytes)
                file_subject = str(email_message["subject"] or "")
                body = email_message.get_body(preferencelist=("plain", "html"))
                file_text = body.get_content() if body is not None else ""

            else:

                file_subject = ""
                file_text = file_bytes.decode("utf-8", errors="ignore")

            with st.expander("Preview Uploaded File Content"):

                if file_subject:
                    st.text(f"Subject: {file_subject}")

                st.text(file_text[:1000])

            if st.button(
                "Analyze Uploaded File",
                type="primary",
                key="btn_file"
            ):

                if not file_subject.strip() and not file_text.strip():

                    st.warning("This file has no text to analyze.")

                else:

                    process_request(
                        "email",
                        {
                            "subject": file_subject,
                            "message": file_text
                        },
                        uploaded_file.name,
                        confidence_threshold
                    )


# --------------------------------------------------
# SCAN HISTORY
# --------------------------------------------------

st.divider()

st.subheader("📜 Scan History")

if not st.session_state.history:

    st.info(
        "No inputs analyzed yet during this session."
    )

else:

    st.dataframe(
        st.session_state.history,
        use_container_width=True
    )