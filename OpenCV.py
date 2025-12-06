import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(1)  # open webcam

model = YOLO("yolo11n.pt")



while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow("Detection", annotated_frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
