import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model = joblib.load("models/random_forest_baseline.pkl")
test = pd.read_csv("data/processed/test_landmarks.csv")

X_test = test.drop(columns=["label"]).to_numpy(dtype="float64")
y_test = test["label"].to_numpy()

predictions = model.predict(X_test)
labels = sorted(test["label"].unique())

print("RANDOM FOREST SECOND-SIGNER RESULTS")
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
print("CONFUSION MATRIX")
print(labels)
print(confusion_matrix(y_test, predictions, labels=labels))