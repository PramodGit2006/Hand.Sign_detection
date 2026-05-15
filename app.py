from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import joblib
import mediapipe as mp

app = Flask(__name__)
CORS(app)

# Load the trained Scikit-Learn model
try:
    model = joblib.load('model.pkl')
    print("Model loaded successfully.")
except Exception as e:
    print(f"WARNING: Could not load model.pkl: {e}")
    model = None

# Initialize MediaPipe Hands precisely as we did in data collection
# Because we are processing sparse video frames from AJAX, static_image_mode=True is more robust
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the base64 encoded image from the JSON payload (from Javascript)
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided', 'prediction': None})
            
        # Front end canvas format: "data:image/jpeg;base64,......."
        image_data = data['image'].split(',')[1]
        
        # Decode the base64 string into raw bytes
        decoded_data = base64.b64decode(image_data)
        np_arr = np.frombuffer(decoded_data, np.uint8)
        
        # Decode image array to OpenCV image
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Extremely Important: Flip it horizontally so it strictly matches 
        # how data_collection.py processed the skeleton before predicting
        frame = cv2.flip(frame, 1)

        # Convert to RGB for MediaPipe parsing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract hand landmarks using our local Google MediaPipe engine
        results = hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return jsonify({'error': 'No hand detected', 'prediction': None})
            
        # Extract features from the active hand
        hand_landmarks = results.multi_hand_landmarks[0]
        wrist = hand_landmarks.landmark[0]
        wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z
        
        normalized_landmarks = []
        for landmark in hand_landmarks.landmark:
            normalized_landmarks.extend([
                landmark.x - wrist_x,
                landmark.y - wrist_y,
                landmark.z - wrist_z
            ])
            
        # Scale Normalization to make it distance-invariant
        max_val = max(abs(val) for val in normalized_landmarks)
        if max_val > 0:
            normalized_landmarks = [val / max_val for val in normalized_landmarks]
            
        # Reshape the 63 floats back into a 2D matrix shape of (1, 63) for Scikit-Learn
        features = np.array(normalized_landmarks).reshape(1, -1)
        
        # Instantly run it through our Neural Network logic!
        if model is None:
            return jsonify({'error': 'Model not loaded on server', 'prediction': None})
        prediction = model.predict(features)[0]
        
        return jsonify({
            'prediction': str(prediction),
            'success': True
        })
        
    except Exception as e:
        print(f"Error predicting: {e}")
        return jsonify({'error': str(e), 'prediction': None})

if __name__ == '__main__':
    # Run locally on accessible Flask development server
    app.run(host='0.0.0.0', port=5000, debug=True)
