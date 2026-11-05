import time
import streamlit as st
from PIL import Image
import io


st.set_page_config(page_title="Spam Detector - Group 8", page_icon="🛡️", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []


def mock_analyzing_input(modality: str, payload: any, source_name: str = "Direct Input") -> dict:
    """
    Placeholder interface for the backend.
    Later, we will replace the body of this function with our actual router/model call.
    """
    start_time = time.perf_counter()

    time.sleep(0.6)

    suspicious_keywords = ["urgent", "winner", "free", "click", "verify", "http://", "password"]
    text_preview = str(payload).lower()

    is_suspicious = any(word in text_preview for word in suspicious_keywords)
    latency_ms = round((time.perf_counter() - start_time) * 1000, 1)

    return {
        "source": source_name,
        "modality": modality,
        "prediction": "Spam" if is_suspicious else "Ham",
        "is_spam": is_suspicious,
        "confidence": 0.9 if is_suspicious else 0.94,
        "latency_ms": latency_ms
    }


def display_result(result: dict):
    """
    Displays the result of the spam detection.
    """
    st.divider()
    st.subheader("Analysis Result")

    if result["is_spam"]:
        st.error(f"⚠️ **Classification: {result['prediction']}**")
    else:
        st.success(f"✅ **Classification: {result['prediction']}**")

    col1, col2, col3 = st.columns(3)
    col1.metric("Detected Modality", result["modality"])
    col2.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
    col3.metric("Latency", f"{result['latency_ms']} ms")


with st.sidebar:
    st.header("⚙️ Prototype Controls")
    st.caption("SE Project Group 8 - UI Prototype")

    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.5,
        max_value=0.99,
        value=0.75,
        step=0.05,
        help="Minimum confidence level required to flag an input",
    )

    st.divider()
    if st.button("🗑️ Clear Scan History", use_container_width=True):
        st.session_state.history = []
        st.rerun()


st.title("🛡️ Multi-Modal Spam Detector")
st.write("Submit text, images or URLs to check for spam content.")

tab_text, tab_url, tab_file = st.tabs(["📝 Text / Email Body", "🔗 URL Check", "📂 File / Image Upload"])

with tab_text:
    user_text = st.text_area(
        "Paste SMS or Email content here:",
        height = 150,
        placeholder="e.g., URGENT: Your account password expires today. Click here to verify...",    
    )
    if st.button("Analyze Text", type="primary", key="btn_text"):
        if not user_text.strip():
            st.warning("Please enter some text before analyzing.")
        else:
            with st.spinner("Routing text to classifier..."):
                res = mock_analyzing_input("Text", user_text, source_name="Pasted Text")
                st.session_state.history.insert(0, res)
            display_result(res)

with tab_url:
    user_url = st.text_input(
        "Enter a URL to inspect:",
        placeholder="e.g., http://example.com/suspicious-link",
    )
    if st.button("Analyze URL", type="primary", key="btn_url"):
        if not user_url.strip():
            st.warning("Please enter a URL before analyzing.")
        else:
            with st.spinner("Routing URL to classifier..."):
                res = mock_analyzing_input("URL", user_url, source_name="Entered URL")
                st.session_state.history.insert(0, res)
            display_result(res)

with tab_file:
    uploaded_file = st.file_uploader(
        "Upload an email (.eml or .txt) or image (.png, .jpg, .jpeg)",
        type=["eml", "txt", "png", "jpg", "jpeg"],
    )
    if uploaded_file is not None:
        file_type = uploaded_file.type

        if "image" in file_type:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Uploaded: {uploaded_file.name}", width=350)
            modality = "Image"
            payload = uploaded_file.name
        else:
            string_data = io.StringIO(uploaded_file.getvalue().decode("utf-8", errors="ignore")).read()
            with st.expander("Preview Uploaded File Content"):
                st.text(string_data[:1000])
            modality = "Email / Document"
            payload = string_data

        if st.button("Analyze Uploaded File", type="primary", key="btn_file"):
            with st.spinner(f"Processing {modality}..."):
                res = mock_analyzing_input(modality, payload, source_name=uploaded_file.name)
                st.session_state.history.insert(0, res)
            display_result(res)


st.divider()
st.subheader("📜 Scan History")

if not st.session_state.history:
    st.info("No inputs analyzed yet during this session.")
else:
    st.dataframe(st.session_state.history, use_container_width=True)

