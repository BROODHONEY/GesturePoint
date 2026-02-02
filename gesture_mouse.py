import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import math

pyautogui.FAILSAFE = False

# Get screen size
screen_width, screen_height = pyautogui.size()

# Smoothing
smoothing = 7
prev_x, prev_y = 0, 0

# Click detection variables
click_threshold = 0.04  # Distance threshold for pinch (experiment with this!)
clicking = False
click_cooldown = 0

# Create hand landmarker with higher confidence for smoother detection
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.8,  # Higher for smoother detection
    min_hand_presence_confidence=0.8,
    min_tracking_confidence=0.8
)
landmarker = vision.HandLandmarker.create_from_options(options)

# For drawing
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

print("Gesture Mouse Control started!")
print("- Move index finger to control mouse")
print("- Pinch thumb and index finger together to click")
print("- Press 'q' to quit")

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two landmarks"""
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

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
            
            # Get important landmarks
            index_tip = hand_landmarks[8]   # Index finger tip
            thumb_tip = hand_landmarks[4]   # Thumb tip
            
            # Convert to screen coordinates
            x = int(index_tip.x * screen_width)
            y = int(index_tip.y * screen_height)
            
            # Smooth movement
            curr_x = prev_x + (x - prev_x) / smoothing
            curr_y = prev_y + (y - prev_y) / smoothing
            
            # Move mouse
            pyautogui.moveTo(curr_x, curr_y, duration=0)
            prev_x, prev_y = curr_x, curr_y
            
            # Calculate distance between thumb and index finger
            distance = calculate_distance(thumb_tip, index_tip)
            
            # Detect pinch gesture for clicking
            if distance < click_threshold:
                if not clicking and click_cooldown == 0:
                    pyautogui.click()
                    clicking = True
                    click_cooldown = 10  # Prevent multiple clicks
                    cv2.putText(frame, "CLICK!", (10, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                clicking = False
            
            # Cooldown counter
            if click_cooldown > 0:
                click_cooldown -= 1
            
            # Visual feedback
            # Draw line between thumb and index
            thumb_px = (int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0]))
            index_px = (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0]))
            
            # Color changes based on distance (red when close = clicking)
            color = (0, 0, 255) if distance < click_threshold else (0, 255, 0)
            cv2.line(frame, thumb_px, index_px, color, 3)
            
            # Display info
            cv2.putText(frame, f"Distance: {distance:.3f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Threshold: {click_threshold}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('Gesture Mouse', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()