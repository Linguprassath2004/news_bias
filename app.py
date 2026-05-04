import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer

# 1. Cache the embedder and model to avoid reloading on every interaction
@st.cache_resource
def load_resources():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    model = joblib.load('models/model.pkl')
    return embedder, model

# 2. Page Configuration
st.set_page_config(
    page_title="News Bias Classifier",
    page_icon="📰",
    layout="centered"
)

st.title("📰 News Bias Classifier")
st.write("Enter a news headline to determine its predicted media bias: **Left, Center, or Right**.")

# 3. Load resources
try:
    with st.spinner("Loading model and embedder..."):
        embedder, model = load_resources()
except Exception as e:
    st.error("Model not found. Please train and save your model using `main.py` first.")
    st.stop()

# 4. User Input Interface
user_input = st.text_input(
    "Enter News Headline or Text:", 
    placeholder="Type or paste the news headline here..."
)

if st.button("Classify Bias", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a valid text input.")
    else:
        with st.spinner("Analyzing bias..."):
            # Generate Embedding
            embedding = embedder.encode([user_input])
            
            # Predict
            prediction = model.predict(embedding)[0]
            
            # Mapping
            label_map = {0: "Left", 1: "Center", 2: "Right"}
            result = label_map[prediction]
            
            # Display output
            st.success("Analysis Complete!")
            st.write(f"### Predicted Bias: **{result}**")
            
            # Optional: Show confidence/probabilities if using a model that supports predict_proba
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(embedding)[0]
                st.write("#### Confidence Scores:")
                st.json({
                    "Left": f"{probabilities[0]*100:.2f}%",
                    "Center": f"{probabilities[1]*100:.2f}%",
                    "Right": f"{probabilities[2]*100:.2f}%"
                })