# app.py
import base64
from fastapi import Path
import streamlit as st
import requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

API_URL = "http://127.0.0.1:8000/predict"   #for local testing
import fitz  

#sidebar

st.set_page_config(page_title="LITMUS", page_icon="logo.png", layout="wide")

with st.sidebar:
    #logo
    def load_logo(path, size=180):
        try:
            img_bytes = open(path, "rb").read()
            encoded = base64.b64encode(img_bytes).decode()
            st.markdown(
                f"""
                <div style='text-align:center; margin-bottom:20px;'>
                    <img src="data:image/png;base64,{encoded}" width="{size}">
                </div>
                """,
                unsafe_allow_html=True
            )
        except:
            st.warning("logo.png not found")

    load_logo("logo.png")




st.markdown("<h1 style='text-align:center;'>LITMUS</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Paste text below to analyze whether it was written by an AI or a human.</p>", unsafe_allow_html=True)

def extract_text_from_pdf(uploaded_file):
    """Extracts text from an uploaded PDF file."""
    try:
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""

        for page in doc:
            text += page.get_text()

        doc.close()
        return text.strip()
    except Exception as e:
        return None


def extract_text_from_txt(uploaded_file):
    """Reads text from a .txt file."""
    try:
        return uploaded_file.read().decode("utf-8").strip()
    except:
        return None
    
def plot_chunk_confidences(probs):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(probs))

    ax.bar(x, probs)
    ax.axhline(0.5, color='red', linestyle='--', label='AI threshold (0.5)')
    ax.set_xlabel("Chunk Index")
    ax.set_ylabel("AI Probability")
    ax.set_title("Chunk-Level AI Probability")
    ax.legend()

    st.pyplot(fig)

def display_colored_chunks(chunks, probs, threshold=0.5):
    st.write("### Chunk-by-chunk Interpretation")
    for i, (chunk, p) in enumerate(zip(chunks, probs)):
        color = "rgba(255, 120, 120, 0.4)" if p > threshold else "rgba(120, 255, 120, 0.4)"
        st.markdown(
            f"""
            <div style="background-color:{color}; padding: 10px; margin-bottom: 10px; border-radius:5px;">
                <b>Chunk {i} — AI Probability: {p:.2f}</b><br>
                {chunk}
            </div>
            """,
            unsafe_allow_html=True
        )

def show_chunk_table(chunks, probs):
    df = pd.DataFrame({
        "chunk_index": list(range(len(chunks))),
        "prob_AI": probs,
        "label": ["AI" if p > 0.5 else "Human" for p in probs]
    })
    st.dataframe(df)


text_area_input = st.text_area("Enter text manually or upload a PDF", height=250)

# final output
final_text = text_area_input

if "uploaded_text" in st.session_state:
    final_text = st.session_state["uploaded_text"]

st.header("Upload PDF or TXT file")
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt"])

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = extract_text_from_txt(uploaded_file)

    if not text:
        st.error("Could not extract text from the file.")
    else:
        st.success("Text extracted successfully!")
        st.write("### Preview of extracted text:")
        st.write(text[:1000] + ".....")   
        
        st.session_state["uploaded_text"] = text

if st.button("Analyze Document"):
        if not final_text.strip():
            st.warning("Please enter or upload text.")
            st.stop()

        with st.spinner("Analyzing..."):
            response = requests.post(API_URL, data={"text": final_text})
            if response.status_code != 200:
                st.error("Backend error. Is FastAPI running?")
            else:
                result = response.json()

                label = result["label"]
                confidence = result["confidence"]
                probs = result["chunk_probs"]
                chunks = result["chunks"]

                #display results
                
                st.header("Overall Prediction")
                # st.write(f"**Label:** {label}")
                # st.write(f"**Confidence:** {confidence:.3f}")
                if(result['label']=="Human-written"):
                    st.success(f"**Label:** {result['label']}") 
                else:
                    st.error(f"**Label:** {result['label']}") 
                st.metric(label="Confidence", value=f"{result['confidence']*100:.2f}%")
                tab1, tab2, tab3 = st.tabs(["Confidence Plot", "Per Chunk Summary", "All Chunks"])
                with tab1:
                    # plot
                    st.subheader("Chunk Confidence Plot")
                    plot_chunk_confidences(probs)
                with tab2:
                    # table
                    st.subheader("Chunk Table")
                    show_chunk_table(chunks, probs)
                with tab3:
                    #colored chunks
                    display_colored_chunks(chunks, probs)

                    # download JSON report, uncomment if seems like a useful feature
                    # import json
                    # st.download_button(
                    #     "Download Detection Report",
                    #     data=json.dumps(result, indent=2),
                    #     file_name="detection_report.json",
                    #     mime="application/json"
                    # )

      