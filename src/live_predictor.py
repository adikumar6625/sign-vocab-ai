import cv2
import time
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp
from collections import defaultdict, deque

DATA_FILE = "data/processed/landmarks.csv"
MODEL_FILE = "models/constructive_prototypes.pkl"
MODEL_PATH = "hand_landmarker.task"
THRESHOLD = 0.25

# Load training data
df = pd.read_csv(DATA_FILE)
X = df.drop(columns=["label"]).to_numpy(dtype="float64")
y = df["label"].to_numpy()

# Build constructive prototypes
prototypes = defaultdict(list)

for label in sorted(set(y)):
    for sample in X[y == label]:
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

# Save the selected model
joblib.dump(
    {
        "prototypes": dict(prototypes),
        "threshold": THRESHOLD
    },
    MODEL_FILE
)

print("Model saved:", MODEL_FILE)

# MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

def predict_sample(sample):
    distances = {}

    for label, class_prototypes in prototypes.items():
        distances[label] = min(
            np.linalg.norm(sample - prototype)
            for prototype in class_prototypes
        )

    predicted_label = min(distances, key=distances.get)
    nearest_distance = distances[predicted_label]

    if nearest_distance > 0.35:
        return "unknown", 0.0

    confidence = max(0, 1 - nearest_distance)
    return predicted_label, confidence

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam.")
    exit()

prediction_history = deque(maxlen=7)
last_timestamp = 0

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        success, frame = cap.read()

        if not success:
            print("Could not read webcam frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(time.time() * 1000)

        if timestamp_ms <= last_timestamp:
            timestamp_ms = last_timestamp + 1

        last_timestamp = timestamp_ms

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        label_text = "Show a gesture"
        confidence_text = ""

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            wrist = hand[0]

            sample = []

            for landmark in hand:
                sample.extend([
                    landmark.x - wrist.x,
                    landmark.y - wrist.y,
                    landmark.z - wrist.z
                ])

                x = int(landmark.x * frame.shape[1])
                y_pixel = int(landmark.y * frame.shape[0])
                cv2.circle(
                    frame,
                    (x, y_pixel),
                    4,
                    (0, 255, 0),
                    -1
                )

            sample = np.array(sample, dtype="float64")

            prediction, confidence = predict_sample(sample)
            prediction_history.append(prediction)

            # Stabilize output using majority voting
            stable_prediction = max(
                set(prediction_history),
                key=prediction_history.count
            )

            label_text = stable_prediction
            confidence_text = f"distance score: {confidence:.2f}"

        cv2.rectangle(
            frame,
            (0, 0),
            (frame.shape[1], 100),
            (20, 20, 20),
            -1
        )

        cv2.putText(
            frame,
            f"Prediction: {label_text}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            confidence_text,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

        cv2.imshow("Incremental Sign Vocabulary", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()