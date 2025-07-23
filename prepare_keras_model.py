import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load dataset
df = pd.read_csv("final_gesture_data.csv")

# Drop rows with NaNs (very important!)
df = df.dropna()

# Features and labels
X = df.drop("label", axis=1).values
y = df["label"].values

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

# Define model
model = Sequential([
    Dense(256, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

# Save model
model.save("gesture_model_keras.h5")
print("[INFO] Saved Keras model as gesture_model_keras.h5")

# Save labels
with open("gesture_labels.txt", "w") as f:
    for label in le.classes_:
        f.write(f"{label}\n")
print("[INFO] Saved gesture_labels.txt")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("gesture_model.tflite", "wb") as f:
    f.write(tflite_model)
print("[INFO] Converted to gesture_model.tflite")
