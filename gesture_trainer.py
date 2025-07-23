import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# === Constants ===
VIDEO_DIR = "video"
FINAL_CSV = "final_gesture_data.csv"
MODEL_FILE = "gesture_model.pkl"
LABEL_FILE = "gesture_labels.txt"

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False)
print("[INFO] MediaPipe Holistic initialized.")

csv_files = []
for filename in os.listdir(VIDEO_DIR):
    if filename.endswith(".mp4"):
        gesture_label = os.path.splitext(filename)[0]
        video_path = os.path.join(VIDEO_DIR, filename)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open: {filename}")
            continue

        all_features = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            features = []

            # Pose landmarks (33 * 4)
            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    features.extend([lm.x, lm.y, lm.z, lm.visibility])
            else:
                features.extend([0] * 33 * 4)

            # Face landmarks (468 * 3)
            # if results.face_landmarks:
            #     for lm in results.face_landmarks.landmark:
            #         features.extend([lm.x, lm.y, lm.z])
            # else:
            #     features.extend([0] * 468 * 3)

            # Left hand (21 * 3)
            if results.left_hand_landmarks:
                for lm in results.left_hand_landmarks.landmark:
                    features.extend([lm.x, lm.y, lm.z])
            else:
                features.extend([0] * 21 * 3)

            # Right hand (21 * 3)
            if results.right_hand_landmarks:
                for lm in results.right_hand_landmarks.landmark:
                    features.extend([lm.x, lm.y, lm.z])
            else:
                features.extend([0] * 21 * 3)

            if any(features):
                all_features.append(features)

        cap.release()

        if all_features:
            df = pd.DataFrame(all_features)
            df["label"] = gesture_label
            csv_name = f"{gesture_label}.csv"
            df.to_csv(csv_name, index=False)
            csv_files.append(csv_name)
            print(f"[INFO] Saved: {csv_name}")
        else:
            print(f"[WARN] No valid frames found in {filename}")

# Merge all CSVs
if not csv_files:
    print("[ERROR] No gesture videos processed. Exiting.")
    exit()

merged_df = pd.concat([pd.read_csv(f) for f in csv_files])
merged_df.to_csv(FINAL_CSV, index=False)
print(f"[INFO] Merged CSV saved: {FINAL_CSV}")

# Train model
X = merged_df.drop("label", axis=1)
y = merged_df["label"]
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"[INFO] Model Accuracy: {acc * 100:.2f}%")

# Save model and labels
joblib.dump(model, MODEL_FILE)
with open(LABEL_FILE, "w") as f:
    for label in le.classes_:
        f.write(f"{label}\n")

print("[SAVED] Model and labels saved.")
