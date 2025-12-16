import streamlit as st
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
import torch

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Political Speech Classifier",
    page_icon="🏛️",
    layout="centered"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
MODEL_PATH = "models/roberta_1980s_paragraph_filtered_epoch3"

@st.cache_resource
def load_model():
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_PATH)
    model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# -----------------------------
# HEADER
# -----------------------------
st.title("🏛️ Political Speech Classifier Demo")
st.info("Model was trained exclusively on speeches from **1981–1989 (97th–100th Congress)**.")

speech = st.text_area(
    "Enter speech text:",
    placeholder="Type a Congressional-style political speech excerpt...",
    height=200
)

# -----------------------------
# RUN BUTTON
# -----------------------------
if st.button("Run Classification"):
    if len(speech.strip()) == 0:
        st.warning("Please enter text before running classification.")
    else:
        inputs = tokenizer(
            speech,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[0]
            pred = torch.argmax(probs).item()

        party = "Democrat (D)" if pred == 0 else "Republican (R)"
        score = float(probs[pred])

        color = "blue" if pred == 0 else "red"

        # -----------------------------
        # CLEAN RESULT OUTPUT
        # -----------------------------
        st.subheader("Prediction Result")
        st.markdown(f"### **<span style='color:{color};'>{party}</span>**", unsafe_allow_html=True)
        st.write(f"Confidence Score: **{score:.4f}**")

        st.progress(score)
