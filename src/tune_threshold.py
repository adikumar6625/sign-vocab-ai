import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("data/processed/landmarks.csv")

X = df.drop(columns=["label"]).to_numpy(dtype="float64")
y = df["label"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

def train_constructive(threshold, class_names):
    prototypes = defaultdict(list)

    for label in class_names:
        samples = X_train[y_train == label]

        for sample in samples:
            if not prototypes[label]:
                prototypes[label].append(sample.copy())
                continue

            distances = [
                np.linalg.norm(sample - p)
                for p in prototypes[label]
            ]

            nearest_index = np.argmin(distances)

            if distances[nearest_index] > threshold:
                prototypes[label].append(sample.copy())
            else:
                prototypes[label][nearest_index] = (
                    0.9 * prototypes[label][nearest_index]
                    + 0.1 * sample
                )

    return prototypes

def predict(prototypes, samples):
    predictions = []

    for sample in samples:
        class_distances = {}

        for label, class_prototypes in prototypes.items():
            class_distances[label] = min(
                np.linalg.norm(sample - p)
                for p in class_prototypes
            )

        predictions.append(min(class_distances, key=class_distances.get))

    return np.array(predictions)

classes = ["hello", "yes", "no", "help", "stop"]
thresholds = [0.20, 0.25, 0.30, 0.35, 0.40]

print("\nTHRESHOLD COMPARISON")

for threshold in thresholds:
    prototypes = train_constructive(threshold, classes)
    predictions = predict(prototypes, X_test)
    accuracy = accuracy_score(y_test, predictions)
    total_prototypes = sum(
        len(items) for items in prototypes.values()
    )

    print(
        f"Threshold: {threshold:.2f} | "
        f"Accuracy: {accuracy:.3f} | "
        f"Prototypes: {total_prototypes}"
    )

best_threshold = max(
    thresholds,
    key=lambda t: accuracy_score(
        y_test,
        predict(
            train_constructive(t, classes),
            X_test
        )
    )
)

best_model = train_constructive(best_threshold, classes)
best_predictions = predict(best_model, X_test)

print("\nBEST THRESHOLD:", best_threshold)
print("FINAL TEST RESULTS")
print(classification_report(y_test, best_predictions))

print("PROTOTYPE COUNTS")
for label, items in best_model.items():
    print(label, len(items))