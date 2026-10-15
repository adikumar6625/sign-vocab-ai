import pandas as pd
import numpy as np
import joblib
from collections import defaultdict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

TRAIN_FILE = "data/processed/landmarks.csv"
TEST_FILE = "data/processed/test_landmarks.csv"
THRESHOLD = 0.25

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

X_train = train.drop(columns=["label"]).to_numpy(dtype="float64")
y_train = train["label"].to_numpy()

X_test = test.drop(columns=["label"]).to_numpy(dtype="float64")
y_test = test["label"].to_numpy()

prototypes = defaultdict(list)

# Build prototypes using only the original signer’s data.
for label in sorted(set(y_train)):
    for sample in X_train[y_train == label]:
        if not prototypes[label]:
            prototypes[label].append(sample.copy())
            continue

        distances = [
            np.linalg.norm(sample - prototype)
            for prototype in prototypes[label]
        ]

        nearest_index = int(np.argmin(distances))

        if distances[nearest_index] > THRESHOLD:
            prototypes[label].append(sample.copy())
        else:
            prototypes[label][nearest_index] = (
                0.9 * prototypes[label][nearest_index]
                + 0.1 * sample
            )

def predict(sample):
    distances = {
        label: min(
            np.linalg.norm(sample - prototype)
            for prototype in class_prototypes
        )
        for label, class_prototypes in prototypes.items()
    }

    return min(distances, key=distances.get)

predictions = np.array([predict(sample) for sample in X_test])

print("SECOND-SIGNER RESULTS")
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

labels = sorted(set(y_test))

print("CONFUSION MATRIX")
print(labels)
print(confusion_matrix(y_test, predictions, labels=labels))

print("\nPROTOTYPE COUNTS")
for label, items in prototypes.items():
    print(label, len(items))