import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import math
import time

pyautogui.FAILSAFE = False

# Screen size
screen_width, screen_height = pyautogui.size()

# IMPROVED SMOOTHING - Exponential Moving Average
alpha = 0.45  # Lower = smoother but slower, Higher = faster but jittery (try 0.2-0.5)
prev_x, prev_y = screen_width // 2, screen_height // 2

# Frame reduction zone (ignore edges where hand might go out)
# This creates a "dead zone" at edges to prevent jumpy behavior
frame_reduction = 0  # 10% margin on each side

# Gesture thresholds
pinch_threshold = 0.05
right_click_threshold = 0.05

# State
is_dragging = False
is_right_clicking = False
click_cooldown = 0

# FPS tracking
prev_time = 0

# Create hand landmarker with OPTIMIZED settings
base_options = python.BaseOptions(model_asset_path='./model/hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,  # Lowered for faster initial detection
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,  # Lowered for smoother tracking
    running_mode=vision.RunningMode.VIDEO  # VIDEO mode is faster than IMAGE mode
)
landmarker = vision.HandLandmarker.create_from_options(options)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Webcam with optimizations
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Lower resolution = faster processing
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)  # Request higher FPS if camera supports it

print("Optimized Gesture Mouse Control!")
print("=" * 50)
print("GESTURES:")
print("- Index finger: Move cursor")
print("- Pinch (Thumb + Index): Click/Drag")
print("- Thumb + Middle finger: Right-click")
print("- Press 'q' to quit")
print("=" * 50)

def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue
    
    frame_count += 1
    
    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
    prev_time = curr_time
    
    frame = cv2.flip(frame, 1)
    
    # Skip RGB conversion and use BGR directly (slight speed improvement)
    # MediaPipe can handle BGR with proper format specification
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Detect with timestamp for VIDEO mode
    results = landmarker.detect_for_video(mp_image, int(curr_time * 1000))
    
    gesture_text = "No hand"
    
    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            # OPTIONAL: Skip drawing landmarks for better performance
            # Comment this section out if you want max FPS
            from mediapipe.framework.formats import landmark_pb2
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) 
                for lm in hand_landmarks
            ])
            
            # mp_drawing.draw_landmarks(
            #     frame,
            #     hand_landmarks_proto,
            #     mp_hands.HAND_CONNECTIONS,
            #     mp_drawing_styles.get_default_hand_landmarks_style(),
            #     mp_drawing_styles.get_default_hand_connections_style()
            # )
            
            # Get landmarks
            index_tip = hand_landmarks[8]
            thumb_tip = hand_landmarks[4]
            middle_tip = hand_landmarks[12]
            
            # Check if hand is within safe zone (not at edges)
            if (frame_reduction < index_tip.x < 1 - frame_reduction and 
                frame_reduction < index_tip.y < 1 - frame_reduction):
                
                # Remap coordinates to account for dead zone
                # This makes the usable area map to full screen
                x_remapped = (index_tip.x - frame_reduction) / (1 - 2 * frame_reduction)
                y_remapped = (index_tip.y - frame_reduction) / (1 - 2 * frame_reduction)
                
                # Convert to screen coordinates
                x = int(x_remapped * screen_width)
                y = int(y_remapped * screen_height)
                
                # Clamp to screen boundaries
                x = max(0, min(screen_width - 1, x))
                y = max(0, min(screen_height - 1, y))
                
                # Exponential Moving Average (EMA) smoothing - MUCH better than division
                curr_x = alpha * x + (1 - alpha) * prev_x
                curr_y = alpha * y + (1 - alpha) * prev_y
                
                # Move mouse
                pyautogui.moveTo(int(curr_x), int(curr_y))
                
                # Update previous
                prev_x, prev_y = curr_x, curr_y
            
            # Gesture detection
            thumb_index_dist = calculate_distance(thumb_tip, index_tip)
            thumb_middle_dist = calculate_distance(thumb_tip, middle_tip)
            
            # Right-Click
            if thumb_middle_dist < right_click_threshold:
                if not is_right_clicking and click_cooldown == 0:
                    pyautogui.rightClick()
                    is_right_clicking = True
                    click_cooldown = 10
                    gesture_text = "RIGHT CLICK!"
                else:
                    gesture_text = "Right clicking..."
            else:
                is_right_clicking = False
            
            # Left-Click & Drag
            if thumb_index_dist < pinch_threshold and not is_right_clicking:
                if not is_dragging:
                    pyautogui.mouseDown()
                    is_dragging = True
                gesture_text = "DRAGGING"
            else:
                if is_dragging:
                    pyautogui.mouseUp()
                    is_dragging = False
                    click_cooldown = 8
                    gesture_text = "Released"
                else:
                    gesture_text = "Moving"
            
            if click_cooldown > 0:
                click_cooldown -= 1
    
    # Display info
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"{gesture_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Alpha: {alpha} (press +/- to adjust)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.imshow('Optimized Gesture Mouse', frame)
    
    # Keyboard controls for live adjustments
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+') or key == ord('='):
        alpha = min(1.0, alpha + 0.05)
        print(f"Alpha increased to {alpha:.2f} (faster, less smooth)")
    elif key == ord('-') or key == ord('_'):
        alpha = max(0.1, alpha - 0.05)
        print(f"Alpha decreased to {alpha:.2f} (slower, more smooth)")

cap.release()
cv2.destroyAllWindows()
landmarker.close()