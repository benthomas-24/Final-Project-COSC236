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
from threading import Thread, Lock
try:
    import select
    SELECT_AVAILABLE = True
except ImportError:
    SELECT_AVAILABLE = False
import signal

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

# --- DISPLAY DETECTION ---
def has_display():
    """
    Check if a display is available for OpenCV windows.
    Returns True if display is available, False otherwise.
    """
    # Check DISPLAY environment variable (Linux/X11)
    if platform.system() == "Linux":
        display = os.environ.get('DISPLAY')
        if not display or display == '':
            return False
        # Try to create a test window to confirm display works
        try:
            cv2.namedWindow("__test__", cv2.WINDOW_NORMAL)
            cv2.destroyWindow("__test__")
            return True
        except:
            return False
    # For macOS (Darwin), assume display is available if we're not in headless mode
    elif platform.system() == "Darwin":
        # Check if we're in a headless environment
        display = os.environ.get('DISPLAY')
        if display == '':
            return False
        try:
            cv2.namedWindow("__test__", cv2.WINDOW_NORMAL)
            cv2.destroyWindow("__test__")
            return True
        except:
            return False
    # Default: assume display available
    return True

# --- CLASS: Keyboard Input Handler (for headless mode) ---
class KeyboardInputHandler:
    """
    Handles keyboard input in headless mode by reading from stdin.
    Reads lines and checks if user typed 'q' followed by Enter.
    """
    def __init__(self):
        self.quit_requested = False
        self.thread = None
        self.running = False
    
    def _read_input(self):
        """Thread function to read keyboard input from stdin."""
        while self.running:
            try:
                # Use select to check if stdin has data (non-blocking)
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    line = sys.stdin.readline().strip().lower()
                    if line == 'q' or line.startswith('q'):
                        self.quit_requested = True
                        break
            except (EOFError, OSError):
                # stdin closed or not available
                break
            except Exception:
                pass
    
    def start(self):
        """Start the keyboard input thread."""
        if not sys.stdin.isatty():
            # Not a TTY, can't read keyboard input interactively
            return False
        if not SELECT_AVAILABLE:
            # select module not available, can't do non-blocking reads
            return False
        self.running = True
        self.thread = Thread(target=self._read_input, daemon=True)
        self.thread.start()
        return True
    
    def stop(self):
        """Stop the keyboard input thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)
    
    def should_quit(self):
        """Check if quit was requested."""
        return self.quit_requested

# --- CLASS: AI Inference Loop (Decoupled) ---
class AIInferenceLoop:
    """
    Runs YOLO inference in a separate thread to decouple AI processing from video recording.
    This prevents the AI from blocking the main video capture loop.
    """
    def __init__(self, model_path="yolo11n.pt"):
        self.model = YOLO(model_path)
        self.stopped = False
        self.input_frame = None
        self.results = None
        self.lock = Lock()  # Crucial for thread safety
        self.inference_fps = 0.0

    def start(self):
        """Start the inference thread."""
        t = Thread(target=self.loop, args=(), daemon=True)
        t.start()
        return self

    def update_frame(self, frame):
        """
        Main thread calls this to give us work.
        MUST copy, otherwise main thread overwrites memory while we are reading.
        """
        with self.lock:
            self.input_frame = frame.copy()

    def get_results(self):
        """
        Main thread calls this to get answers.
        Returns (results, inference_fps) tuple.
        """
        with self.lock:
            return self.results, self.inference_fps

    def loop(self):
        """Main inference loop running in background thread."""
        while not self.stopped:
            # 1. Safely grab image
            frame_to_process = None
            with self.lock:
                if self.input_frame is not None:
                    frame_to_process = self.input_frame.copy()
            
            # 2. Run AI (The slow part)
            if frame_to_process is not None:
                start = time.time()
                
                # imgsz=320 is the "Happy Medium" for Pi 5 speed vs accuracy
                results = self.model.predict(frame_to_process, imgsz=320, conf=0.25, verbose=False)
                
                fps = 1.0 / (time.time() - start) if (time.time() - start) > 0 else 0.0
                
                # 3. Safely store results
                with self.lock:
                    self.results = results
                    self.inference_fps = fps
            else:
                time.sleep(0.01)  # Don't burn CPU if no frame

    def stop(self):
        """Stop the inference thread."""
        self.stopped = True

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

# Initialize AI Inference Loop (runs in separate thread)
print("Loading YOLO model...")
ai = AIInferenceLoop("yolo11n.pt").start() 

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


# UI Constants (Base values - will be scaled based on frame dimensions)
FONT = cv2.FONT_HERSHEY_SIMPLEX
BASE_FONT_SCALE = 0.6
BASE_LINE_HEIGHT = 25
BASE_PADDING = 10
BASE_MARGIN = 15  # Margin from edges
TEXT_COLOR = (255, 255, 255)
BOX_COLOR = (0, 0, 0)

def calculate_ui_dimensions(frame_width, frame_height, stats_lines):
    """
    Calculate UI dimensions dynamically based on frame size.
    Returns a dict with all UI positioning and sizing values.
    """
    # Scale factors based on frame dimensions (normalized to 1920x1080 reference)
    ref_width = 1920
    ref_height = 1080
    width_scale = frame_width / ref_width
    height_scale = frame_height / ref_height
    # Use average scale for consistent sizing
    scale = (width_scale + height_scale) / 2.0
    
    # Scale font and spacing based on frame size
    font_scale = BASE_FONT_SCALE * scale
    line_height = int(BASE_LINE_HEIGHT * scale)
    padding = int(BASE_PADDING * scale)
    margin = int(BASE_MARGIN * scale)
    
    # Calculate maximum text width by measuring all lines
    max_text_width = 0
    for line in stats_lines:
        if line:
            (text_w, _), _ = cv2.getTextSize(line, FONT, font_scale, 2)
            max_text_width = max(max_text_width, text_w)
    
    # UI box dimensions (text area + padding)
    ui_width = max_text_width + (padding * 2)
    ui_height = len(stats_lines) * line_height + (padding * 2)
    
    # Position in bottom-right corner with margins
    text_x = frame_width - ui_width - margin
    text_y_start = frame_height - ui_height - margin
    
    # Ensure UI doesn't go off screen
    text_x = max(margin, text_x)
    text_y_start = max(margin, text_y_start)
    
    # Box coordinates (extend padding around text)
    box_x1 = text_x - padding
    box_y1 = text_y_start - padding
    box_x2 = text_x + max_text_width + padding
    box_y2 = text_y_start + (len(stats_lines) * line_height) + padding
    
    return {
        'font_scale': font_scale,
        'line_height': line_height,
        'padding': padding,
        'margin': margin,
        'ui_width': ui_width,
        'ui_height': ui_height,
        'text_x': text_x,
        'text_y_start': text_y_start,
        'box_x1': box_x1,
        'box_y1': box_y1,
        'box_x2': box_x2,
        'box_y2': box_y2
    } 

# FPS Tracking
frame_count = 0
start_time = time.time()
current_fps = 0.0  # Initialize FPS display value
fps_window = []  # For smoothing FPS display

# Check if display is available
DISPLAY_AVAILABLE = has_display()
if not DISPLAY_AVAILABLE:
    print("No display detected. Running in headless mode (video will not be displayed).")

# Initialize keyboard input handler for headless mode
keyboard_handler = None
if not DISPLAY_AVAILABLE:
    keyboard_handler = KeyboardInputHandler()
    if keyboard_handler.start():
        print("Keyboard input enabled. Press 'q' and Enter to stop.")
    else:
        print("Note: Keyboard input not available. Use Ctrl+C to stop.")

print(f"Recording to: {output_filename}")
print(f"Device: {DEVICE_NAME}")
print(f"Target FPS: {TARGET_FPS}")
if DISPLAY_AVAILABLE:
    print("Press 'q' to stop.")

# Variable to remember the LAST known detection
latest_ai_results = None
latest_inference_fps = 0.0

try:
    while True:
        loop_start = time.time()  # Mark start of loop

        # 1. READ (Non-blocking now!)
        frame = vs.read()
        if frame is None:
            break
        
        # Make a copy for annotation to avoid threading race conditions
        annotated_frame = frame.copy()

        # 2. HAND OFF FRAME TO AI THREAD (Non-blocking)
        ai.update_frame(frame)

        # 3. GET RESULTS FROM AI THREAD (Use latest available)
        new_results, ai_fps = ai.get_results()
        if new_results is not None:
            latest_ai_results = new_results
            latest_inference_fps = ai_fps

        # 4. MANUAL DRAWING (Faster than .plot())
        # Use latest_ai_results even if they are slightly old
        if latest_ai_results and len(latest_ai_results) > 0 and len(latest_ai_results[0].boxes) > 0:
            for box in latest_ai_results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{ai.model.names[cls]} {conf:.2f}"
                
                # Draw Box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw Label Background
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
                
                # Draw Label Text
                cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 4. STATS UI (only draw if display is available)
        cpu_usage = get_cpu_usage()
        mem_percent, mem_string = get_memory_usage()

        # Get GPU usage
        gpu_usage = get_gpu_usage()
        
        # Only draw UI overlay if display is available
        if DISPLAY_AVAILABLE:
            stats_lines = [
                DEVICE_NAME,
                "",
                f"CPU: {cpu_usage:.1f}%" if cpu_usage is not None else "CPU: N/A",
                f"GPU: {gpu_usage:.1f}%" if gpu_usage is not None else "GPU: N/A",
                f"Mem: {mem_percent:.1f}% ({mem_string})" if mem_percent is not None and mem_string else "Mem: N/A",
                f"FPS: {current_fps:.1f}",
                f"AI FPS: {latest_inference_fps:.1f}" if latest_inference_fps > 0 else "AI FPS: N/A"
            ]

            # Calculate UI dimensions dynamically based on frame size
            ui_dims = calculate_ui_dimensions(vs.width, vs.height, stats_lines)

            # Draw UI Background (Dynamic position/size)
            cv2.rectangle(
                annotated_frame,
                (ui_dims['box_x1'], ui_dims['box_y1']),
                (ui_dims['box_x2'], ui_dims['box_y2']),
                BOX_COLOR,
                -1
            )

            # Draw UI Text
            y_offset = ui_dims['text_y_start']
            for i, line in enumerate(stats_lines):
                if line:
                    # Device name slightly larger
                    scale = ui_dims['font_scale'] + (0.1 * (vs.width / 1920)) if i == 0 else ui_dims['font_scale']
                    thickness = 3 if i == 0 else 2
                    
                    cv2.putText(annotated_frame, line, (ui_dims['text_x'], y_offset), 
                              FONT, scale, TEXT_COLOR, thickness, cv2.LINE_AA)
                y_offset += ui_dims['line_height']

        # 5. DISPLAY & WRITE
        if DISPLAY_AVAILABLE:
            cv2.imshow("Recording - Detection", annotated_frame)
            if cv2.waitKey(1) == ord('q'):
                break
        
        # Write frame to video file (always, regardless of display)
        out.write(annotated_frame)
        frame_count += 1
        
        # Check for quit in headless mode
        if not DISPLAY_AVAILABLE and keyboard_handler and keyboard_handler.should_quit():
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
        # Calculate inference latency from AI FPS
        inference_ms = (1000.0 / latest_inference_fps) if latest_inference_fps > 0 else None
        
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
ai.stop()
out.release()
if DISPLAY_AVAILABLE:
    cv2.destroyAllWindows()
if keyboard_handler:
    keyboard_handler.stop()

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
