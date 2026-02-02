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

# Smoothing
alpha = 0.45
prev_x, prev_y = screen_width // 2, screen_height // 2

# Frame reduction (dead zone)
frame_reduction = 0

# Gesture thresholds
pinch_threshold = 0.05
right_click_threshold = 0.05

# State
is_dragging = False
is_right_clicking = False
click_cooldown = 0

# FPS tracking
prev_time = 0
fps_history = []

# Create hand landmarker
base_options = python.BaseOptions(model_asset_path='./model/hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.VIDEO
)
landmarker = vision.HandLandmarker.create_from_options(options)

# Webcam
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("=" * 60)
print("GESTURE MOUSE CONTROL")
print("=" * 60)
print("CONTROLS:")
print("  • Move cursor: Point with index finger")
print("  • Click/Drag: Pinch thumb + index finger")
print("  • Right-click: Pinch thumb + middle finger")
print()
print("KEYBOARD SHORTCUTS:")
print("  • Toggle visualization: Press 'v'")
print("  • Increase smoothing: Press '+'")
print("  • Decrease smoothing: Press '-'")
print("  • Quit: Press 'q'")
print("=" * 60)

def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

# Visualization toggle
show_visualization = False
frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue
    
    frame_count += 1
    
    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
    prev_time = curr_time
    
    fps_history.append(fps)
    if len(fps_history) > 30:
        fps_history.pop(0)
    avg_fps = sum(fps_history) / len(fps_history)
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    results = landmarker.detect_for_video(mp_image, int(curr_time * 1000))
    
    gesture_text = "No hand detected"
    hand_detected = False
    
    if results.hand_landmarks:
        hand_detected = True
        for hand_landmarks in results.hand_landmarks:
            
            # Get landmarks
            index_tip = hand_landmarks[8]
            thumb_tip = hand_landmarks[4]
            middle_tip = hand_landmarks[12]
            
            # MOUSE MOVEMENT
            if (frame_reduction < index_tip.x < 1 - frame_reduction and 
                frame_reduction < index_tip.y < 1 - frame_reduction):
                
                # Remap coordinates
                x_remapped = (index_tip.x - frame_reduction) / (1 - 2 * frame_reduction)
                y_remapped = (index_tip.y - frame_reduction) / (1 - 2 * frame_reduction)
                
                # Convert to screen coordinates
                x = int(x_remapped * screen_width)
                y = int(y_remapped * screen_height)
                
                # Clamp to screen boundaries
                x = max(0, min(screen_width - 1, x))
                y = max(0, min(screen_height - 1, y))
                
                # EMA smoothing
                curr_x = alpha * x + (1 - alpha) * prev_x
                curr_y = alpha * y + (1 - alpha) * prev_y
                
                # Move mouse
                pyautogui.moveTo(int(curr_x), int(curr_y))
                prev_x, prev_y = curr_x, curr_y
            
            # GESTURE DETECTION
            thumb_index_dist = calculate_distance(thumb_tip, index_tip)
            thumb_middle_dist = calculate_distance(thumb_tip, middle_tip)
            
            # Right-Click: Thumb + Middle finger
            if thumb_middle_dist < right_click_threshold:
                if not is_right_clicking and click_cooldown == 0:
                    pyautogui.rightClick()
                    is_right_clicking = True
                    click_cooldown = 10
                    gesture_text = "RIGHT CLICK"
                else:
                    gesture_text = "Right clicking..."
            else:
                is_right_clicking = False
            
            # Left-Click & Drag: Thumb + Index finger
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
            
            # Cooldown
            if click_cooldown > 0:
                click_cooldown -= 1
            
            # VISUALIZATION (optional)
            if show_visualization:
                frame_h, frame_w = frame.shape[:2]
                thumb_px = (int(thumb_tip.x * frame_w), int(thumb_tip.y * frame_h))
                index_px = (int(index_tip.x * frame_w), int(index_tip.y * frame_h))
                middle_px = (int(middle_tip.x * frame_w), int(middle_tip.y * frame_h))
                
                # Draw fingertip circles
                cv2.circle(frame, thumb_px, 8, (0, 255, 0), -1)
                cv2.circle(frame, index_px, 8, (255, 0, 0), -1)
                cv2.circle(frame, middle_px, 8, (255, 255, 0), -1)
                
                # Draw lines between fingers
                color_ti = (0, 0, 255) if thumb_index_dist < pinch_threshold else (0, 255, 0)
                color_tm = (255, 0, 255) if thumb_middle_dist < right_click_threshold else (255, 255, 0)
                cv2.line(frame, thumb_px, index_px, color_ti, 2)
                cv2.line(frame, thumb_px, middle_px, color_tm, 2)
    
    # UI Display
    fps_color = (0, 255, 0) if avg_fps > 20 else (0, 255, 255) if avg_fps > 15 else (0, 0, 255)
    
    cv2.putText(frame, f"FPS: {int(avg_fps)}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
    cv2.putText(frame, gesture_text, (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Smoothing: {alpha:.2f}", (10, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.imshow('Gesture Mouse Control', frame)
    
    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('v'):
        show_visualization = not show_visualization
        print(f"Visualization: {'ON' if show_visualization else 'OFF'}")
    elif key == ord('+') or key == ord('='):
        alpha = min(1.0, alpha + 0.05)
        print(f"Smoothing (alpha): {alpha:.2f}")
    elif key == ord('-') or key == ord('_'):
        alpha = max(0.1, alpha - 0.05)
        print(f"Smoothing (alpha): {alpha:.2f}")

cap.release()
cv2.destroyAllWindows()
landmarker.close()

print(f"\nSession ended. Average FPS: {int(sum(fps_history) / len(fps_history))}")