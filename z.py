import json
import cv2
import numpy as np
import streamlit as st
import google.generativeai as genai

from htr_pipeline import (
    read_page,
    DetectorConfig,
    LineClusteringConfig,
    ReaderConfig,
    PrefixTree
)

# --------------------------------------------------
# Load dictionary
# --------------------------------------------------
# Ensure data/words_alpha.txt exists in your directory
try:
    with open("data/words_alpha.txt") as f:
        word_list = [w.strip().upper() for w in f.readlines()]
    prefix_tree = PrefixTree(word_list)
except FileNotFoundError:
    st.error("Dictionary file not found. Please ensure 'data/words_alpha.txt' exists.")
    word_list = []
    prefix_tree = None

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Detect and Read Handwritten Words",
    layout="wide"
)

st.title("📝 Detect and Read Handwritten Words")

# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------
st.sidebar.header("Parameters")

scale = st.sidebar.slider("Scale", 0.0, 10.0, 1.0, step=0.01)
margin = st.sidebar.slider("Margin", 0, 25, 1)
use_dictionary = st.sidebar.checkbox("Use dictionary", value=True)
min_words_per_line = st.sidebar.slider("Minimum words per line", 1, 10, 2)
text_scale = st.sidebar.slider("Text size in visualization", 0.5, 2.0, 1.0)

st.sidebar.markdown("---")
st.sidebar.header("AI Refinement")
api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Get your key from aistudio.google.com")

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def refine_text_with_gemini(raw_text, key):
    """
    Sends raw OCR text to Gemini for correction.
    """
    if not key:
        return None
    
    try:
        genai.configure(api_key=key)
        # using gemini-1.5-flash for speed
        model = genai.GenerativeModel("gemini-2.5-flash") 
        
        prompt = (
            "You are an expert at correcting text extracted from handwritten notes. "
            "Please correct the grammar, spelling, and punctuation of the following text. "
            "Do not add any conversational filler. Just output the corrected text.\n\n"
            f"Raw Text:\n{raw_text}"
        )
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error connecting to Gemini: {e}"

def process_page(img_bgr):
    if prefix_tree is None and use_dictionary:
        st.error("Dictionary not loaded. Cannot use dictionary decoding.")
        return "", img_bgr

    read_lines = read_page(
        img_bgr,
        detector_config=DetectorConfig(scale=scale, margin=margin),
        line_clustering_config=LineClusteringConfig(
            min_words_per_line=min_words_per_line
        ),
        reader_config=ReaderConfig(
            decoder="word_beam_search" if use_dictionary else "best_path",
            prefix_tree=prefix_tree
        ),
    )

    # Extract text
    text_output = ""
    for read_line in read_lines:
        text_output += " ".join(w.text for w in read_line) + "\n"

    # Draw bounding boxes
    vis_img = img_bgr.copy()
    for read_line in read_lines:
        for read_word in read_line:
            aabb = read_word.aabb
            cv2.rectangle(
                vis_img,
                (aabb.xmin, aabb.ymin),
                (aabb.xmin + aabb.width, aabb.ymin + aabb.height),
                (255, 0, 0),
                2,
            )
            cv2.putText(
                vis_img,
                read_word.text,
                (aabb.xmin, aabb.ymin + aabb.height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                (255, 0, 0),
                2,
            )

    return text_output, vis_img

# --------------------------------------------------
# Run pipeline
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload handwritten image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)

    if st.button("🚀 Run Handwriting Recognition"):
        with st.spinner("Processing image (HTR)..."):
            raw_text, vis = process_page(img)

        # Display Visualization
        with col2:
            st.subheader("Visualization")
            st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_column_width=True)

        # Display Raw Output
        st.subheader("📄 Raw Recognized Text")
        st.text_area("Raw Output", raw_text, height=150)

        # AI Refinement Section
        st.markdown("---")
        st.subheader("✨ Gemini Refined Text")

        if api_key:
            with st.spinner("Refining text with Gemini..."):
                refined_text = refine_text_with_gemini(raw_text, api_key)
                if refined_text:
                    st.text_area("Refined Output", refined_text, height=150)
                else:
                    st.error("Failed to get a response from Gemini.")
        else:
            st.info("ℹ️ Enter your Gemini API Key in the sidebar to enable AI text correction and formatting.")






# AIzaSyAQDb0fvkX0SUxPJ-uNfOfIEUL8gPONp8E
