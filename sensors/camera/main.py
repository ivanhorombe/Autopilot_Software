
import os
import cv2
import time
import numpy as np
from vision import ArUcoTracker
from controllers import PIDController, EMAFilter, Odometer
from mission import MissionManager
from display import Dashboard
from hardware import KartHardware

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
# ==========================================
# --- HYPERPARAMETERS (Tune these!) ---
# ==========================================
# Vision
M_SIZE = 0.164         # Real marker width in meters
CALIB_FILE = 'calibration_data.npz'

# Control Gains
STEER_KP = 0.04        # Steering aggression
STEER_KD = 0.01        # Steering damping
S_SMOOTH = 0.15        # Steering filter (lower = smoother)
V_SMOOTH = 0.20        # Velocity filter

# Safety/Mission
STOP_DIST = 1.0        # Hard stop distance (m)
WARN_DIST = 4.0        # Slow down distance (m)
TIMEOUT = 0.5          # Signal loss timeout (s)

SERIAL_PORT = 'COM4'  # Change to '/dev/ttyUSB0' on Linux/Mac
# ==========================================

# --- INITIALIZE ---
hardware = KartHardware(port=SERIAL_PORT)

# --- INSIDE THE WHILE LOOP (Section 3: ACT) ---
# After you calculate final_steer and curr_throttle:

# --- INITIALIZE MODULES ---
tracker = ArUcoTracker(CALIB_FILE, marker_size=M_SIZE)
steer_pid = PIDController(kp=STEER_KP, kd=STEER_KD)
steer_filter = EMAFilter(alpha=S_SMOOTH)
speedometer = Odometer(alpha=V_SMOOTH)
mission = MissionManager()
ui = Dashboard()

cap = cv2.VideoCapture(1)
prev_time = time.time()
last_seen_time = time.time()
velocity = 0.0  # Placeholder until you add speed sensors
# Persistent state
curr_steer = 0.0
curr_throttle = 0.0
# --- HARDWARE CONFIGURATION ---
CHOSEN_CAMERA = 0      # Try 0, 1, or 2 if your camera isn't opening
SERIAL_PORT = 'COM4'   # This must match Device Manager for the ESP32
cap = cv2.VideoCapture(CHOSEN_CAMERA)

if not cap.isOpened():
    print(f"--- ERROR: Camera {CHOSEN_CAMERA} not found! ---")
    print("Try changing CHOSEN_CAMERA to 1 or 2.")
    exit()
else:
    print(f"--- SUCCESS: Camera {CHOSEN_CAMERA} is online ---")
while True:
    ret, frame = cap.read()
    if not ret: break
    
    curr_time = time.time()
    dt = curr_time - prev_time
    
    # 1. SENSE
    m_id, x, z, tvec, corners = tracker.get_target(frame)
    
    # 2. THINK
    if m_id is not None:
        last_seen_time = curr_time
        
        # Update Physics & Logic
        velocity = speedometer.update(z, dt)
        mode, max_thr, bias = mission.process_id(m_id, z)
        
        # Calculate Smooth Steering
        angle_err = np.degrees(np.arctan2(x, z))
        raw_target = steer_pid.compute(angle_err, dt) + bias
        curr_steer = steer_filter.apply(raw_target)
        
        # Speed Safety logic
        if z < STOP_DIST:
            curr_throttle = 0.0
            msg, color = "COLLISION AVOIDANCE", (0, 0, 255)
        elif z < WARN_DIST:
            curr_throttle = max_thr * 0.4
            msg, color = "CAUTION: APPROACHING", (0, 165, 255)
        else:
            curr_throttle = max_thr
            msg, color = "OPTIMAL PATH", (0, 255, 0)
            
        # Change the last number to 0.05 (5cm) or 0.03 (3cm)
        cv2.drawFrameAxes(frame, tracker.cam_matrix, tracker.dist_coeffs, np.zeros(3), tvec, 0.05)
    else:
        # HEARTBEAT (Signal Loss)
        if (curr_time - last_seen_time) > TIMEOUT:
            curr_throttle = 0.0
            curr_steer = 0.0
            mode, msg, color = "FAILSAFE", "NO SIGNAL", (0, 0, 255)
            velocity = 0.0 # Odometer can't calculate without vision
            z = None
    hardware.send_command(curr_steer, curr_throttle)

    # 3. ACT (Display output)
    ui.draw(frame, velocity, curr_steer, curr_throttle, mode, msg, color, dist=z)
    
    prev_time = curr_time
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
#python main.py