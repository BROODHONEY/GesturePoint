import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import numpy as np

# Disable PyAutoGUI failsafe (moving mouse to corner won't stop program)
pyautogui.FAILSAFE = False

# Get screen size
screen_width, screen_height = pyautogui.size()
print(f"Screen size: {screen_width}x{screen_height}")

# Smoothing variables (reduces jitter)
smoothing = 3
prev_x, prev_y = 0, 0

# Create hand landmarker
base_options = python.BaseOptions(model_asset_path='./model/hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
landmarker = vision.HandLandmarker.create_from_options(options)

# For drawing
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Start webcam
cap = cv2.VideoCapture(1, cv2.CAP_MSMF)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Mouse control started!")
print("- Move your index finger to control the mouse")
print("- Press 'q' to quit")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    results = landmarker.detect(mp_image)
    
    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            # Draw landmarks
            from mediapipe.framework.formats import landmark_pb2
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) 
                for lm in hand_landmarks
            ])
            
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Get index finger tip (landmark 8)
            index_tip = hand_landmarks[8]
            
            # Convert normalized coordinates (0-1) to screen coordinates
            # Normalized: 0-1, Screen: 0-screen_width or 0-screen_height
            x = int(index_tip.x * screen_width)
            y = int(index_tip.y * screen_height)
            
            # Smoothing to reduce jitter
            # This averages current position with previous positions
            curr_x = prev_x + (x - prev_x) / smoothing
            curr_y = prev_y + (y - prev_y) / smoothing
            
            # Move mouse
            pyautogui.moveTo(curr_x, curr_y, duration=0)
            
            # Update previous position
            prev_x, prev_y = curr_x, curr_y
            
            # Show coordinates on screen
            cv2.putText(frame, f"Mouse: ({int(curr_x)}, {int(curr_y)})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Mouse Controller', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()