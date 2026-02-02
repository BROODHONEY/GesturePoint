import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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
cap = cv2.VideoCapture(0)

print("Hand tracking started! Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue
    
    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Detect hands
    results = landmarker.detect(mp_image)
    
    # Draw landmarks
    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            # Convert to drawing format
            landmark_list = []
            for landmark in hand_landmarks:
                landmark_list.append(landmark)
            
            # Draw on frame
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
            
            # Print index finger tip position for debugging
            index_tip = hand_landmarks[8]
            cv2.putText(frame, f"Index: ({index_tip.x:.2f}, {index_tip.y:.2f})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Hand Tracking - MediaPipe Tasks', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()