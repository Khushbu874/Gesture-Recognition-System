from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)

interpreter = tf.lite.Interpreter(model_path="gesture_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open("gesture_labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = list(map(float, data.values()))  # Convert dict to list of float values

        if len(features) != input_details[0]['shape'][1]:
            return jsonify({'label': 'Invalid Input Length', 'confidence': 0.0})

        input_data = np.array([features], dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        class_index = int(np.argmax(output_data[0]))
        confidence = float(np.max(output_data[0]))
        label = labels[class_index]

        print(f"Gesture: {label}, Confidence: {confidence}")
        return jsonify({'label': label, 'confidence': confidence})

    except Exception as e:
        print("Error:", e)
        return jsonify({'label': 'Error', 'confidence': 0.0, 'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
