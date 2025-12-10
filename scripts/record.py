import cv2
from ultralytics import YOLO
import os
import sys
import time
import platform
import socket
import re
import subprocess
import shutil
import argparse  # NEW
from threading import Thread

# Add project root to Python path to import utils
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.metrics import MetricsLogger, RunMetadata  # NEW

# --- CONFIGURATION ---
TARGET_FPS = 30.0  # Set video to standard 30 FPS
FRAME_TIME = 1.0 / TARGET_FPS  # How long one frame should take (0.033s)

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description='Record video with YOLO object detection and metrics logging')
parser.add_argument('--tag', type=str, default='run', help='Tag for this recording run (used in metrics filename)')
parser.add_argument('--sample-every', type=int, default=1, help='Sample metrics every N frames (default: 1)')
args = parser.parse_args()

# Try to import psutil for system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not installed. Install with 'pip install psutil'.")

# Try to import GPU monitoring libraries
GPU_AVAILABLE = False
GPU_LIB = None
GPU_HANDLE = None
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
    GPU_LIB = "pynvml"
    GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except:
    try:
        import GPUtil
        GPU_AVAILABLE = True
        GPU_LIB = "GPUtil"
    except:
        GPU_AVAILABLE = False

# --- CLASS: Threaded Video Capture ---
class WebcamStream:
    """
    Reads frames in a separate thread to reduce latency.
    """
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            raise ValueError("Camera not accessible")
            
        # Set buffer size to 1 to reduce internal OpenCV buffering
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Get properties for the writer
        self.width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# --- HELPER: System Stats ---
def get_device_name():
    uname = platform.uname()
    node = uname.node
    system = uname.system
    
    # Mac Model
    if system == "Darwin":
        try:
            mac_model = subprocess.check_output(["/usr/sbin/sysctl", "-n", "hw.model"], encoding="utf-8").strip()
            if mac_model == "Mac16,7": return "M4 Pro Macbook Pro"
            return mac_model
        except: pass

    # RPi Model
    if system == "Linux":
        try:
            with open("/proc/device-tree/model", "r") as f:
                rpi_model = f.read().strip('\x00').strip()
                if "Raspberry Pi 5" in rpi_model: return "Raspberry Pi 5"
                return rpi_model
        except: pass

    return node if node else socket.gethostname()

def sanitize_for_filename(name):
    return re.sub(r'[^a-zA-Z0-9-_]+', '_', name)

def get_cpu_usage():
    return psutil.cpu_percent(interval=None) if PSUTIL_AVAILABLE else None

def get_memory_usage():
    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        return mem.percent, f"{mem.used/(1024**3):.1f}GB:{mem.total/(1024**3):.1f}GB"
    return None, None

def get_gpu_usage():
    if not GPU_AVAILABLE: return None
    try:
        if GPU_LIB == "pynvml":
            return pynvml.nvmlDeviceGetUtilizationRates(GPU_HANDLE).gpu
        elif GPU_LIB == "GPUtil":
            gpus = GPUtil.getGPUs()
            if gpus: return gpus[0].load * 100
    except: pass
    return None

def get_recording_directory():
    """
    Determines the recording directory based on device type.
    Returns: directory path string
    """
    uname = platform.uname()
    system = uname.system
    
    # Check if it's a MacBook (Darwin system)
    if system == "Darwin":
        return "macbook-recordings"
    
    # Check if it's a Raspberry Pi (Linux system with Raspberry Pi in model)
    if system == "Linux":
        try:
            with open("/proc/device-tree/model", "r") as f:
                rpi_model = f.read().strip('\x00').strip()
                if "Raspberry Pi" in rpi_model:
                    return "RaspberryPi-Recordings"
        except:
            pass
    
    # Default: create a generic recordings directory
    return "recordings"

def get_unique_filename(base_path, base_name):
    """
    Generate a unique filename in the specified directory.
    Creates directory if it doesn't exist.
    """
    # Create directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    # Full path
    full_path = os.path.join(base_path, base_name)
    
    if not os.path.exists(full_path):
        return full_path
    
    # If file exists, find next available number
    name, ext = os.path.splitext(base_name)
    k = 1
    while os.path.exists(os.path.join(base_path, f"{name}{k}{ext}")):
        k += 1
    return os.path.join(base_path, f"{name}{k}{ext}")

def select_video_input():
    """
    Detects available video input devices and prompts user to select one.
    Returns the selected device index.
    """
    print("\nScanning for available video input devices...")
    available_devices = []
    
    # Test devices 0-9 (most systems won't have more than this)
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Try to read a frame to confirm it's working
            ret, _ = cap.read()
            if ret:
                # Try to get device name/info
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                device_info = f"Device {i} ({width}x{height}"
                if fps > 0:
                    device_info += f", {fps} FPS"
                device_info += ")"
                available_devices.append((i, device_info))
            cap.release()
    
    if not available_devices:
        print("No video input devices found!")
        return None
    
    # Display available devices
    print("\nAvailable video input devices:")
    print("-" * 50)
    for idx, (device_num, info) in enumerate(available_devices):
        print(f"  [{idx}] {info}")
    print("-" * 50)
    
    # Get user selection
    while True:
        try:
            selection = input(f"\nSelect device (0-{len(available_devices)-1}): ").strip()
            device_idx = int(selection)
            if 0 <= device_idx < len(available_devices):
                selected_device = available_devices[device_idx][0]
                print(f"Selected: {available_devices[device_idx][1]}\n")
                return selected_device
            else:
                print(f"Please enter a number between 0 and {len(available_devices)-1}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nCancelled.")
            return None

# --- MAIN SETUP ---
DEVICE_NAME = get_device_name()
DEVICE_NAME_SAFE = sanitize_for_filename(DEVICE_NAME)

# Determine recording directory based on device type
recording_dir = get_recording_directory()
base_filename = f'recorded_input_{DEVICE_NAME_SAFE}.mp4'
output_filename = get_unique_filename(recording_dir, base_filename)

# Select video input device
selected_device = select_video_input()
if selected_device is None:
    print("No device selected. Exiting.")
    exit(1)

# Initialize Model
print("Loading YOLO model...")
model = YOLO("yolo11n.pt") 

# Start Stream
print(f"Starting camera stream on device {selected_device}...")
vs = WebcamStream(src=selected_device).start()
time.sleep(1.0) # Allow camera to warm up

# Video Writer Setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_filename, fourcc, TARGET_FPS, (vs.width, vs.height))

# --- METRICS LOGGER (NEW) ---
MODEL_NAME = "yolo11n.pt"
meta = RunMetadata(
    device_name=DEVICE_NAME,
    model_name=MODEL_NAME,
    target_fps=TARGET_FPS,
    resolution=f"{vs.width}x{vs.height}",
)
metrics = MetricsLogger(out_dir="metrics", run_tag=args.tag, meta=meta, sample_every=args.sample_every)


# UI Constants (Defined ONCE for performance)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)
BOX_COLOR = (0, 0, 0)
LINE_HEIGHT = 25
PADDING = 10
PAD_X = 15
PAD_Y = 10

# Pre-calculate UI width to prevent jitter
dummy_long_text = "Memory Usage: 100.0% (16.0GB:16.0GB)"
(text_w, _), _ = cv2.getTextSize(dummy_long_text, FONT, FONT_SCALE, FONT_THICKNESS)
FIXED_UI_WIDTH = text_w + 20 

# FPS Tracking
frame_count = 0
start_time = time.time()
current_fps = 0.0  # Initialize FPS display value
fps_window = []  # For smoothing FPS display

print(f"Recording to: {output_filename}")
print(f"Device: {DEVICE_NAME}")
print(f"Target FPS: {TARGET_FPS}")
print("Press 'q' to stop.")

try:
    while True:
        loop_start = time.time()  # Mark start of loop

        # 1. READ (Non-blocking now!)
        frame = vs.read()
        if frame is None:
            break
        
        # Make a copy for annotation to avoid threading race conditions
        annotated_frame = frame.copy()

        # 2. INFERENCE (TIMED)
        infer_start = time.time()
        results = model.predict(annotated_frame, conf=0.25, verbose=False)
        inference_time = time.time() - infer_start  # seconds
        inference_ms = inference_time * 1000.0

        # 3. MANUAL DRAWING (Faster than .plot())
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{model.names[cls]} {conf:.2f}"
            
            # Draw Box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw Label Background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            
            # Draw Label Text
            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 4. STATS UI
        cpu_usage = get_cpu_usage()
        mem_percent, mem_string = get_memory_usage()

        # Get GPU usage
        gpu_usage = get_gpu_usage()
        
        stats_lines = [
            DEVICE_NAME,
            "",
            f"CPU: {cpu_usage:.1f}%" if cpu_usage is not None else "CPU: N/A",
            f"GPU: {gpu_usage:.1f}%" if gpu_usage is not None else "GPU: N/A",
            f"Mem: {mem_percent:.1f}% ({mem_string})" if mem_percent is not None and mem_string else "Mem: N/A",
            f"FPS: {current_fps:.1f}"
        ]

        # Draw UI Background (Fixed position/size)
        total_height = len(stats_lines) * LINE_HEIGHT
        text_x = vs.width - FIXED_UI_WIDTH - PAD_X
        text_y_start = vs.height - total_height - PAD_Y

        cv2.rectangle(
            annotated_frame,
            (text_x - PADDING, text_y_start - PADDING),
            (vs.width - PAD_X + PADDING, vs.height - PAD_Y + PADDING),
            BOX_COLOR,
            -1
        )

        # Draw UI Text
        y_offset = text_y_start
        for i, line in enumerate(stats_lines):
            if line:
                # Device name slightly larger
                scale = FONT_SCALE + 0.1 if i == 0 else FONT_SCALE
                thickness = FONT_THICKNESS + 1 if i == 0 else FONT_THICKNESS
                
                cv2.putText(annotated_frame, line, (text_x, y_offset), FONT, scale, TEXT_COLOR, thickness, cv2.LINE_AA)
            y_offset += LINE_HEIGHT

        # 5. DISPLAY & WRITE
        cv2.imshow("Recording - Detection", annotated_frame)
        out.write(annotated_frame)
        frame_count += 1

        if cv2.waitKey(1) == ord('q'):
            break

        # --- THE FIX: FPS LIMITER ---
        # Calculate how long processing took
        processing_time = time.time() - loop_start
        
        # Calculate how long we need to sleep to match TARGET_FPS
        delay = FRAME_TIME - processing_time
        
        # If we are too fast, sleep the difference
        if delay > 0:
            time.sleep(delay)
        
        # Calculate ACTUAL loop time (post-sleep) for FPS calculation
        actual_loop_time = time.time() - loop_start
        
        # Calculate FPS based on actual loop time (only if time is reasonable)
        if actual_loop_time > 0.001:  # Avoid division by very small numbers
            frame_fps = 1.0 / actual_loop_time
            # Cap FPS at reasonable maximum (e.g., 1000 FPS) to avoid display issues
            frame_fps = min(frame_fps, 1000.0)
            fps_window.append(frame_fps)
            if len(fps_window) > 10:  # Keep last 10 frames for smoothing
                fps_window.pop(0)
            # Calculate smoothed average FPS
            current_fps = sum(fps_window) / len(fps_window) if fps_window else 0
        else:
            # If loop time is suspiciously small, keep previous FPS value
            pass
        # --- LOG METRICS (NEW) ---
        metrics.log(
            frame=frame_count,
            cpu=cpu_usage if cpu_usage is not None else None,
            gpu=gpu_usage if gpu_usage is not None else None,
            mem=mem_percent if mem_percent is not None else None,
            fps=current_fps,
            processing_time=processing_time,
            delay=delay,
            actual_loop_time=actual_loop_time,
            model_latency_ms=inference_ms,
        )

        # Print fps periodically to debug
        if frame_count % 30 == 0:
            print(f"FPS: {current_fps:.1f} (Target: {TARGET_FPS})")

except KeyboardInterrupt:
    print("\nStopping...")

# --- CLEANUP ---
vs.stop()
out.release()
cv2.destroyAllWindows()

elapsed_total = time.time() - start_time
final_avg_fps = frame_count / elapsed_total if elapsed_total > 0 else 0

# --- SAVE METRICS (NEW) ---
csv_path = metrics.save()

if csv_path:
    print(f"Saved metrics CSV: {csv_path}")
else:
    print("No metrics recorded.")

print(f"Saved: {output_filename}")
print(f"Final Average FPS: {final_avg_fps:.2f}")

# --- FFMPEG ADJUSTMENT (if significant mismatch) ---
if abs(final_avg_fps - TARGET_FPS) > 2.0:
    print(f"Warning: Significant FPS mismatch. Adjusting file to {final_avg_fps:.2f}...")
    if shutil.which("ffmpeg"):
        temp_filename = output_filename.replace('.mp4', '_temp.mp4')
        try:
            cmd = ["ffmpeg", "-y", "-i", output_filename, "-c", "copy", "-r", str(final_avg_fps), temp_filename]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            os.replace(temp_filename, output_filename)
            print("✓ Framerate adjusted.")
        except Exception as e:
            print(f"Adjustment failed: {e}")
            if os.path.exists(temp_filename): os.remove(temp_filename)
    else:
        print("Install ffmpeg for automatic framerate correction.")
