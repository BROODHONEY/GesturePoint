import cv2
import time

print("Starting OpenCV test")

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
print("Camera opened:", cap.isOpened())

time.sleep(2)  # give backend time

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    cv2.imshow("OPENCV TEST", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Exited cleanly")
