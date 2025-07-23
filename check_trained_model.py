import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="gesture_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load gesture labels
with open("gesture_labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False)

# Open test video
cap = cv2.VideoCapture("video/thanks.mp4")  # Replace with your test video

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)

    features = []

    # Pose
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        features.extend([0] * 33 * 4)

    # Left Hand
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0] * 21 * 3)

    # Right Hand
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0] * 21 * 3)

    # Skip if feature length doesn't match
    expected_len = input_details[0]['shape'][1]
    if len(features) != expected_len:
        print(f"[WARNING] Feature length mismatch: expected {expected_len}, got {len(features)}")
        continue

    # Predict
    input_data = np.array([features], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])  # Shape: [1, num_classes]

    class_index = int(np.argmax(output_data[0]))
    confidence = float(np.max(output_data[0]))
    label = labels[class_index]

    # Show result
    cv2.putText(frame, f"{label} ({confidence:.2f})", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.imshow("Gesture Detection (Keras TFLite)", frame)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
