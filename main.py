from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.preprocess import preprocess
from src.embeddings import EmbeddingModel
from src.train import train_model
from src.evaluate import evaluate_model
from src.config import TEST_SIZE, RANDOM_STATE

def main():
    print("STARTING PROGRAM...")

    print("Loading data...")
    df = load_data()

    print("Preprocessing...")
    df = preprocess(df)

    print("Generating embeddings...")
    embedder = EmbeddingModel()
    X = embedder.encode(df['clean_headline'].tolist())
    y = df['label_encoded']

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training model...")
    model = train_model(X_train, y_train)

    print("Evaluating...")
    evaluate_model(model, X_test, y_test)

    print("DONE")

if __name__ == "__main__":
    main()