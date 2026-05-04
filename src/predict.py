import joblib
from src.config import MODEL_PATH

def load_model():
    return joblib.load(MODEL_PATH)

def predict(text, embedder, model):
    embedding = embedder.encode([text])
    pred = model.predict(embedding)[0]

    reverse_map = {0: "left", 1: "neutral", 2: "right"}
    return reverse_map[pred]