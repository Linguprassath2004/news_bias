import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text

def preprocess(df):
    print("Preprocessing data...")

    # Clean text
    df['clean_headline'] = df['text'].apply(clean_text)

    # Normalize labels (important)
    df['label'] = df['label'].str.lower().str.strip()

    # Correct mapping
    label_map = {
        "left": 0,
        "center": 1,
        "right": 2
    }

    df['label_encoded'] = df['label'].map(label_map)

    # Remove any unexpected labels (safety)
    df = df.dropna(subset=['label_encoded'])

    return df