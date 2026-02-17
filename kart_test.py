import os
import cv2
import numpy as np
import tensorflow as tf
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from collections import deque

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 1. LOAD MODEL
MODEL_PATH = "best_kart_model.keras" 
RESOLUTION = (160, 96) # (Width, Height)

if not os.path.exists(MODEL_PATH):
    print(f"FATAL: Put {MODEL_PATH} in this folder before running.")
    exit()

model = tf.keras.models.load_model(MODEL_PATH)

# 2. SETUP BUFFER (Crucial for 9-channel stacking)
frame_buffer = deque(maxlen=3)

# 3. Setup Env
env = MetaDriveEnv(dict(
    use_render=True,
    manual_control=False,
    map=6,
    traffic_density=0.1,
    image_observation=True,
    start_seed=0,
    sensors={
        "rgb_camera": (RGBCamera, RESOLUTION[0], RESOLUTION[1]),
    },
    vehicle_config={
        "image_source": "rgb_camera",
        "length": 2,
        "width": 1.5,
        "height": 1.0,
        "mass": 150,
    },
))

def preprocess_image(rgb_raw):
    # MetaDrive usually returns float [0, 1] or uint8 [0, 255]
    if rgb_raw.dtype != np.uint8:
        img = (rgb_raw[..., :3] * 255).astype(np.uint8)
    else:
        img = rgb_raw[..., :3]
    
    # Resize and normalize
    img = cv2.resize(img, RESOLUTION) 
    return img.astype(np.float32) / 255.0

obs, _ = env.reset()
print("🏁 AI is driving with 9-channel temporal vision...")

try:
    while True:
        # Get sensor data
        rgb_sensor = env.engine.get_sensor("rgb_camera")
        rgb_raw = rgb_sensor.get_image(env.agent)
        
        if len(rgb_raw.shape) == 4:
            rgb_raw = np.squeeze(rgb_raw)

        # 1. Add current frame to rolling buffer
        processed_frame = preprocess_image(rgb_raw)
        frame_buffer.append(processed_frame)

        # 2. Skip until we have 3 frames to stack
        if len(frame_buffer) < 3:
            env.step([0, 0])
            continue

        # 3. Stack frames: [t, t-1, t-2] -> shape (96, 160, 9)
        # Using list(frame_buffer)[::-1] ensures order is (current, prev, older)
        input_stack = np.concatenate(list(frame_buffer)[::-1], axis=-1)
        input_stack = np.expand_dims(input_stack, axis=0)
        
        # ... (Previous prediction code) ...
        
        # 1. Prediction
        predictions = model.predict(input_stack, verbose=0)
        steering = float(predictions[0][0][0])
        pred_mask = predictions[1][0] # Shape (96, 160, 1)

        # 2. FAIL-SAFE LOGIC
        # Calculate what percentage of the image is 'drivable'
        # If the mask is mostly 0s, we are off-track or facing a wall.
        drivable_ratio = np.mean(pred_mask > 0.5) # Percentage of white pixels
        
        if drivable_ratio < 0.02:  # Less than 2% of the view is road
            print(f"⚠️ EMERGENCY BRAKE: Only {drivable_ratio:.1%} road visible!")
            steering = 0.0
            throttle = -1.0  # Slam on the anchors
        else:
            throttle = 0.45  # Normal cruise speed
            
        # 3. VISUALIZATION
        # Create a heat-map style view to see 'confidence'
        mask_viz = cv2.applyColorMap((pred_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Overlay the mask on the current frame for "Augmented Reality" debugging
        current_frame_display = (list(frame_buffer)[-1] * 255).astype(np.uint8)
        overlay = cv2.addWeighted(current_frame_display, 0.7, mask_viz, 0.3, 0)
        
        cv2.imshow("AI Perception Overlay", overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 4. ACT
        obs, reward, terminated, truncated, info = env.step([steering, throttle])
        
        if terminated or truncated:
            frame_buffer.clear() # Clear buffer on reset
            env.reset()

except KeyboardInterrupt:
    print("\nStopping AI test...")
finally:
    env.close()
    cv2.destroyAllWindows()
#python kart_test.py