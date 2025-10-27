import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Safe Social Media Platform", page_icon="🛡️")

st.title("🛡️ Safe Social Media Platform")
st.write("A mini app that detects and flags toxic or abusive comments automatically.")

# Load pre-trained toxicity detection model
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="unitary/toxic-bert")

nlp = load_model()

# Text input
comment = st.text_area("💬 Enter your comment:")

if st.button("Check Comment"):
    if comment.strip() == "":
        st.warning("Please enter a comment.")
    else:
        result = nlp(comment)[0]
        label = result['label']
        score = result['score']

        if label.lower() == "toxic" and score > 0.6:
            st.error(f"⚠️ Toxic Comment Detected! ({score*100:.1f}% confidence)\nPlease rephrase your comment.")
        else:
            st.success("✅ Comment is Safe and Positive!")
            st.write(f"**Your Comment:** {comment}")

st.markdown("---")
st.caption("Developed as a Mini Project – Safe Social Media Platform")
