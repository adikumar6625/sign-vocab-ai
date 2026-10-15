import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from collections import defaultdict
import joblib
import os

df = pd.read_csv("data/processed/landmarks.csv")
X = df.drop("label", axis=1).values.astype(float)
y = df["label"].values

prototypes = defaultdict(list)
THRESHOLD = 0.35

def add_sample(label, sample):
    if not prototypes[label]:
        prototypes[label].append(sample.copy())
        return

    distances = [
        np.linalg.norm(sample - prototype)
        for prototype in prototypes[label]
    ]

    nearest_index = np.argmin(distances)
    nearest_distance = distances[nearest_index]

    if nearest_distance > THRESHOLD:
        prototypes[label].append(sample.copy())
    else:
        prototypes[label][nearest_index] = (
            0.9 * prototypes[label][nearest_index] + 0.1 * sample
        )

def learn_classes(class_names):
    for label in class_names:
        samples = X[y == label]

        for sample in samples:
            add_sample(label, sample)

        print(
            f"Added {label}: "
            f"{len(prototypes[label])} prototypes"
        )

def predict(samples):
    results = []

    for sample in samples:
        distances = {}

        for label, class_prototypes in prototypes.items():
            distances[label] = min(
                np.linalg.norm(sample - prototype)
                for prototype in class_prototypes
            )

        results.append(min(distances, key=distances.get))

    return np.array(results)

initial_classes = ["hello", "yes", "no", "help"]
new_classes = ["stop", "i_love_you", "where", "bathroom"]

learn_classes(initial_classes)

stage_1_mask = np.isin(y, initial_classes)
stage_1_pred = predict(X[stage_1_mask])

print("\nSTAGE 1")
print("Accuracy:", accuracy_score(y[stage_1_mask], stage_1_pred))
print(classification_report(y[stage_1_mask], stage_1_pred))

learn_classes(new_classes)

stage_2_pred = predict(X)

print("\nSTAGE 2")
print("Accuracy:", accuracy_score(y, stage_2_pred))
print(classification_report(y, stage_2_pred))

old_mask = np.isin(y, initial_classes)
old_pred = predict(X[old_mask])

print("OLD-CLASS RETENTION:")
print(accuracy_score(y[old_mask], old_pred))

os.makedirs("models", exist_ok=True)

joblib.dump(
    {
        "prototypes": dict(prototypes),
        "threshold": THRESHOLD
    },
    "models/constructive_prototypes.pkl"
)

print("\nConstructive model saved successfully.")