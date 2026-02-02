import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import math

pyautogui.FAILSAFE = False

# Screen size
screen_width, screen_height = pyautogui.size()

# Smoothing
smoothing = 7
prev_x, prev_y = 0, 0

# Gesture thresholds
pinch_threshold = 0.04      # Thumb + Index = Click
drag_threshold = 0.05       # Thumb + Index held = Drag
right_click_threshold = 0.04  # Thumb + Middle finger = Right click

# State variables
is_clicking = False
is_dragging = False
is_right_clicking = False
click_cooldown = 0

# Create hand landmarker
base_options = python.BaseOptions(model_asset_path='./model/hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.6
)
landmarker = vision.HandLandmarker.create_from_options(options)

# For drawing
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

print("Advanced Gesture Mouse Control!")
print("=" * 50)
print("GESTURES:")
print("- Index finger: Move cursor")
print("- Pinch (Thumb + Index): Click")
print("- Hold pinch: Drag")
print("- Thumb + Middle finger: Right-click")
print("- Press 'q' to quit")
print("=" * 50)

def calculate_distance(point1, point2):
    """Calculate distance between two landmarks"""
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def count_fingers(hand_landmarks):
    """Count how many fingers are up (simple detection)"""
    fingers_up = 0
    
    # Thumb (check if tip is right of IP joint for right hand)
    if hand_landmarks[4].x < hand_landmarks[3].x:
        fingers_up += 1
    
    # Other fingers: check if tip is above PIP joint
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
    finger_pips = [6, 10, 14, 18]
    
    for tip, pip in zip(finger_tips, finger_pips):
        if hand_landmarks[tip].y < hand_landmarks[pip].y:
            fingers_up += 1
    
    return fingers_up

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    results = landmarker.detect(mp_image)
    
    gesture_text = "No hand detected"
    
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
            
            # Get key landmarks
            index_tip = hand_landmarks[8]    # Index finger tip
            thumb_tip = hand_landmarks[4]    # Thumb tip
            middle_tip = hand_landmarks[12]  # Middle finger tip
            
            # Convert to screen coordinates
            x = int(index_tip.x * screen_width)
            y = int(index_tip.y * screen_height)
            
            # Smooth movement
            curr_x = prev_x + (x - prev_x) / smoothing
            curr_y = prev_y + (y - prev_y) / smoothing
            
            # Move mouse
            pyautogui.moveTo(curr_x, curr_y, duration=0)
            prev_x, prev_y = curr_x, curr_y
            
            # Calculate distances
            thumb_index_dist = calculate_distance(thumb_tip, index_tip)
            thumb_middle_dist = calculate_distance(thumb_tip, middle_tip)
            
            # Count fingers up (for future gestures)
            fingers_count = count_fingers(hand_landmarks)
            
            # GESTURE DETECTION
            
            # Right-Click: Thumb + Middle finger pinch
            if thumb_middle_dist < right_click_threshold:
                if not is_right_clicking and click_cooldown == 0:
                    pyautogui.rightClick()
                    is_right_clicking = True
                    click_cooldown = 15
                    gesture_text = "RIGHT CLICK!"
                else:
                    gesture_text = "Right clicking..."
            else:
                is_right_clicking = False
            
            # Left-Click & Drag: Thumb + Index finger pinch
            if thumb_index_dist < pinch_threshold and not is_right_clicking:
                if not is_clicking and click_cooldown == 0:
                    # Start clicking/dragging
                    pyautogui.mouseDown()
                    is_clicking = True
                    is_dragging = True
                    gesture_text = "DRAGGING"
                else:
                    gesture_text = "Dragging..."
            else:
                # Release if was dragging
                if is_dragging:
                    pyautogui.mouseUp()
                    is_dragging = False
                    is_clicking = False
                    click_cooldown = 10
                    gesture_text = "Released"
                else:
                    gesture_text = f"Moving (Fingers: {fingers_count})"
            
            # Cooldown
            if click_cooldown > 0:
                click_cooldown -= 1
            
            # VISUAL FEEDBACK
            frame_h, frame_w = frame.shape[:2]
            
            # Draw line between thumb and index
            thumb_px = (int(thumb_tip.x * frame_w), int(thumb_tip.y * frame_h))
            index_px = (int(index_tip.x * frame_w), int(index_tip.y * frame_h))
            middle_px = (int(middle_tip.x * frame_w), int(middle_tip.y * frame_h))
            
            # Thumb-Index line (for left click/drag)
            color_ti = (0, 0, 255) if thumb_index_dist < pinch_threshold else (0, 255, 0)
            cv2.line(frame, thumb_px, index_px, color_ti, 3)
            
            # Thumb-Middle line (for right click)
            color_tm = (255, 0, 255) if thumb_middle_dist < right_click_threshold else (255, 255, 0)
            cv2.line(frame, thumb_px, middle_px, color_tm, 2)
            
            # Display info
            y_offset = 30
            cv2.putText(frame, f"Gesture: {gesture_text}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30
            cv2.putText(frame, f"Thumb-Index: {thumb_index_dist:.3f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_ti, 2)
            y_offset += 25
            cv2.putText(frame, f"Thumb-Middle: {thumb_middle_dist:.3f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_tm, 2)
            y_offset += 25
            cv2.putText(frame, f"Fingers up: {fingers_count}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    else:
        # No hand detected - display message
        cv2.putText(frame, gesture_text, 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow('Advanced Gesture Mouse', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
landmarker.close()