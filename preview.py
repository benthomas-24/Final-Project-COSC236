import cv2
from ultralytics import YOLO

# Open webcam (same camera as OpenCV.py)
cap = cv2.VideoCapture(0)
model = YOLO("yolo11n.pt")

print("Preview started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLO inference with same parameters as recording script
    results = model.predict(frame, conf=0.25, verbose=False)
    
    # Plot the results with bounding boxes
    annotated_frame = results[0].plot()
    
    # Display the annotated frame
    cv2.imshow("Preview - Detection", annotated_frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Preview closed.")

