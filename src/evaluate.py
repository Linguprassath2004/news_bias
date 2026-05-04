from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test):
    print("Evaluating model...")

    # Predictions
    y_pred = model.predict(X_test)

    # Classification report
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["left", "neutral", "right"]))

    # Confusion matrix (printed as text)
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n")
    print(cm)