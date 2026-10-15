import cv2
import csv
import os
import time
import mediapipe as mp

GESTURES = ["where", "help", "no", "bathroom"]
SAMPLES_PER_GESTURE = 50
OUTPUT_FILE = "data/processed/landmarks.csv"
MODEL_PATH = "hand_landmarker.task"

os.makedirs("data/processed", exist_ok=True)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

header = ["label"]

for i in range(21):
    header += [f"x{i}", f"y{i}", f"z{i}"]

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

file_exists = os.path.exists(OUTPUT_FILE)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam.")
    exit()

try:
    with open(OUTPUT_FILE, "a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(header)

        with HandLandmarker.create_from_options(options) as landmarker:
            timestamp_ms = 0

            for gesture in GESTURES:
                count = 0

                print(f"\nPrepare gesture: {gesture}")
                print("Press SPACE to save a sample. Press Q to quit.")

                while count < SAMPLES_PER_GESTURE:
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

                    timestamp_ms += 1

                    result = landmarker.detect_for_video(
                        mp_image,
                        timestamp_ms
                    )

                    cv2.putText(
                        frame,
                        f"{gesture}: {count}/{SAMPLES_PER_GESTURE}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

                    if result.hand_landmarks:
                        hand = result.hand_landmarks[0]

                        for landmark in hand:
                            x = int(landmark.x * frame.shape[1])
                            y = int(landmark.y * frame.shape[0])

                            cv2.circle(
                                frame,
                                (x, y),
                                4,
                                (0, 255, 0),
                                -1
                            )

                    cv2.imshow("Gesture Data Collection", frame)

                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        print("\nCollection stopped.")
                        raise KeyboardInterrupt

                    if key == 32 and result.hand_landmarks:
                        hand = result.hand_landmarks[0]
                        wrist = hand[0]

                        row = [gesture]

                        for landmark in hand:
                            row.extend([
                                landmark.x - wrist.x,
                                landmark.y - wrist.y,
                                landmark.z - wrist.z
                            ])

                        writer.writerow(row)
                        file.flush()

                        count += 1
                        print(
                            f"{gesture}: "
                            f"saved {count}/{SAMPLES_PER_GESTURE}"
                        )

                print(f"Finished collecting: {gesture}")

except KeyboardInterrupt:
    print("\nCollection stopped by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()

print(f"\nDataset saved to: {OUTPUT_FILE}")