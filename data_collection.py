import cv2
import mediapipe as mp
from mediapipe.python import solutions
import numpy as np
import csv
import os

# Initialize MediaPipe Hands
mp_hands = solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = solutions.drawing_utils

CSV_FILE = "dataset.csv"

# Initialize CSV file with headers if it doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        # label + 21 landmarks * 3 coordinates (x, y, z)
        header = ['label']
        for i in range(21):
            header.extend([f'x{i}', f'y{i}', f'z{i}'])
        writer.writerow(header)

# Attempt to open the webcam
cap = cv2.VideoCapture(0)

print("========================================")
print("       Data Collection Started          ")
print("========================================")
print("1. Stand in front of your webcam.")
print("2. Make a hand sign (0-9).")
print("3. Press the corresponding number key (0-9) to save a frame of coordinates.")
print("Press 'q' or 'ESC' to quit.")
print("========================================")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue

    # Flip the frame horizontally for a more intuitive selfie-view
    frame = cv2.flip(frame, 1)
    
    # Convert the BGR image to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image to find hand landmarks
    results = hands.process(rgb_frame)
    
    normalized_landmarks = []
    
    # If a hand is detected in the frame
    if results.multi_hand_landmarks:
        # We only expect max_num_hands=1, but loop just in case
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the landmarks on the video feed
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
            )
            
            # Get the wrist landmark as the origin (0, 0, 0) for normalization
            wrist = hand_landmarks.landmark[0]
            wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z
            
            # Extract and normalize (shift) coordinates relative to the wrist
            for landmark in hand_landmarks.landmark:
                norm_x = landmark.x - wrist_x
                norm_y = landmark.y - wrist_y
                norm_z = landmark.z - wrist_z
                normalized_landmarks.extend([norm_x, norm_y, norm_z])
                
            # Scale Normalization to make it distance-invariant
            max_val = max(abs(val) for val in normalized_landmarks)
            if max_val > 0:
                normalized_landmarks = [val / max_val for val in normalized_landmarks]
    
    # Display instructions on the screen
    cv2.putText(frame, "Press 0-9 to save label. 'q' to quit.", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Show the video feed with the hand skeleton drawn
    cv2.imshow('Hand Sign Data Collection', frame)
    
    # Wait 1 ms for key press
    key = cv2.waitKey(1) & 0xFF
    
    # Handle user interaction
    if key == 27 or key == ord('q'): # ESC or 'q' to exit loop
        break
    elif ord('0') <= key <= ord('9'): # Keys '0' to '9'
        if len(normalized_landmarks) == 63: # Ensure we have all 21 points * 3 (xyz)
            label = chr(key)
            with open(CSV_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                row = [label] + normalized_landmarks
                writer.writerow(row)
            print(f"✅ Success! Saved hand pose as '{label}'.")
        else:
            print("⚠️ Error: Hand not fully detected. Make sure your whole hand is visible.")

cap.release()
cv2.destroyAllWindows()
