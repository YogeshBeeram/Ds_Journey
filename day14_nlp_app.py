import streamlit as st
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")


@st.cache_resource
def load_models():
    sentiment = pipeline("sentiment-analysis")
    ner = pipeline("ner", aggregation_strategy="simple")
    zero_shot = pipeline("zero-shot-classification")
    return sentiment, ner, zero_shot


st.title("🤖 NLP Text Analyzer")
st.write("Analyze any text using AI — sentiment, entities, and topic detection!")

# Load models
with st.spinner("Loading AI models..."):
    sentiment, ner, zero_shot = load_models()

# User input
text = st.text_area("Enter any text here:", 
    placeholder="Type or paste any text...",
    height=150)

if st.button("Analyze Text 🤖"):
    if text:
        # 1. Sentiment
        st.write("### 🎭 Sentiment Analysis")
        result = sentiment(text)[0]
        if result['label'] == 'POSITIVE':
            st.success(f"POSITIVE 😊 ({result['score']*100:.1f}% confidence)")
        else:
            st.error(f"NEGATIVE 😞 ({result['score']*100:.1f}% confidence)")

        # 2. NER
        st.write("### 🏷️ Named Entities Found")
        entities = ner(text)
        if entities:
            for e in entities:
                st.write(f"**{e['word']}** → {e['entity_group']} ({e['score']*100:.1f}%)")
        else:
            st.write("No named entities found!")

        # 3. Zero Shot
        st.write("### 🎯 Topic Classification")
        labels = ["technology", "sports", "finance", 
                  "politics", "health", "entertainment"]
        result = zero_shot(text, candidate_labels=labels)
        top3 = list(zip(result['labels'][:3], result['scores'][:3]))
        for label, score in top3:
            st.progress(score)
            st.write(f"**{label}**: {score*100:.1f}%")
    else:
        st.warning("Please enter some text first!")

st.write("---")
st.write("Built with HuggingFace Transformers 🤗")

