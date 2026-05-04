from sklearn.linear_model import LogisticRegression
import joblib

MODEL_PATH = "models/model.pkl"

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    return model