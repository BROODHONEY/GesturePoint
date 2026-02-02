import cv2
print("Testing camera...")

cap = cv2.VideoCapture(0)
print("Opened:", cap.isOpened())

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not read")
        break

    cv2.imshow("TEST", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
