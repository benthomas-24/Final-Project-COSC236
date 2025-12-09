'''
- I want to add a way for data and metrics to be recordedd on charts using matplotlib.
- This will be helpful so I can distigush between different hardware limitations from the Raspberry Pi and the M4 Pro Macbook Pro.
- I should document thigns like frame rates, CPU ussage, and memory usage.
- I want to make clean graphs and charts that are easy to understand and visualize for my slideshow.

'''


import cv2
from ultralytics import YOLO
import time
import os

cap = cv2.VideoCapture(0)  # open webcam
model = YOLO("yolo11n.pt")

# Set desired recording FPS (adjust this to control playback speed)
# Lower values = slower playback, Higher values = faster playback
DESIRED_FPS = 30 # Change this to control speed (e.g., 15 for half speed, 30 for normal)

# Function to generate unique filename
def get_unique_filename(base_name):
    """Generate a unique filename by appending a number if file exists."""
    if not os.path.exists(base_name):
        return base_name
    
    # Split filename and extension
    name, ext = os.path.splitext(base_name)
    k = 1
    
    # Find the next available number
    while os.path.exists(f"{name}{k}{ext}"):
        k += 1
    
    return f"{name}{k}{ext}"

# Get unique filename
output_filename = get_unique_filename('recorded_input.mp4')

# Get the video properties for VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Use desired FPS for video writer
out = cv2.VideoWriter(output_filename, fourcc, DESIRED_FPS, (width, height))

# Calculate time per frame to limit capture rate
frame_time = 1.0 / DESIRED_FPS
frame_count = 0
start_time = time.time()

print(f"Recording at {DESIRED_FPS} FPS. Press Ctrl+C to stop.")
print(f"Saving to: {output_filename}")

try:
    while True:
        frame_start = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO inference
        results = model.predict(frame, conf=0.25, verbose=False)
        
        # Plot the results with bounding boxes
        annotated_frame = results[0].plot()

        # Save annotated frame to the video file
        out.write(annotated_frame)
        
        frame_count += 1
        
        # Limit frame rate to match desired FPS
        elapsed = time.time() - frame_start
        sleep_time = frame_time - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        # Print progress every 5 seconds
        if frame_count % (DESIRED_FPS * 5) == 0:
            elapsed_total = time.time() - start_time
            print(f"Recorded {frame_count} frames ({elapsed_total:.1f}s)")
            
except KeyboardInterrupt:
    print("\nStopping recording...")

cap.release()
out.release()
print(f"Video saved to {output_filename}")
