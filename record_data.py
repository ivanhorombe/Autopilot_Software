import os
import cv2
import numpy as np
import pandas as pd
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.sensors.semantic_camera import SemanticCamera
from metadrive.component.sensors.rgb_camera import RGBCamera
import random
random_seed = random.randint(0, 9999)  # Generate a random seed for this session
with open("current_seed.txt", "w") as f:
    f.write(str(random_seed))
# --- SETTINGS ---
SAVE_EVERY = 3 
JPEG_QUALITY = 70 
RESOLUTION = (160, 96) # Width, Height
DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")

if os.path.exists(DATA_DIR):
    import shutil
    shutil.rmtree(DATA_DIR)
os.makedirs(IMAGE_DIR, exist_ok=True)

env = MetaDriveEnv(dict(
    use_render=True,
    manual_control=True,
    map="XSSS",  # Simple map for consistent data
    traffic_density=0.05,
    image_observation=True,
    
    # --- RANDOM SEED ---
    # Setting start_seed=0 (or any int) makes the map and traffic consistent
    start_seed=random_seed,  # Change this for different sessions, or set to None for random each time
    
    sensors={
        "rgb_camera": (RGBCamera, RESOLUTION[0], RESOLUTION[1]),
        "semantic_camera": (SemanticCamera, RESOLUTION[0], RESOLUTION[1])
    },
    
    # --- KART SIZE & CAMERA SOURCE ---
    vehicle_config={
        "image_source": "rgb_camera",
        # Kart-like dimensions (Standard car is approx 4.5m x 1.8m)
        "length": 2,  # Shorter for a kart
        "width": 1.5,   # Narrower
        "height": 1.0,  # Lower profile
        "mass": 150,  # Lighter weight
    },
))

obs, _ = env.reset()
data_log = []
mask_buffer = [] 
saved = 0

print("🚀 HIGH-SPEED COLLECTION - Correct orientation...")

try:
    while True:
        obs, reward, terminated, truncated, info = env.step([0, 0])
        
        # Pull sensors directly
        rgb_sensor = env.engine.get_sensor("rgb_camera")
        semantic_sensor = env.engine.get_sensor("semantic_camera")

        # Capture RGB - NO FLIP, MetaDrive provides correct orientation
        rgb_raw = rgb_sensor.get_image(env.agent)
        
        # Squeeze if needed
        if len(rgb_raw.shape) == 4:
            rgb_raw = np.squeeze(rgb_raw)
        
        # Convert to uint8 and BGR
        if rgb_raw.max() <= 1.0:
            rgb_img = (rgb_raw[..., :3] * 255).astype(np.uint8)
        else:
            rgb_img = rgb_raw[..., :3].astype(np.uint8)
        
        rgb_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

        # Capture Semantic Mask - NO FLIP
        sem_raw = semantic_sensor.get_image(env.agent)
        
        # Squeeze if needed
        if len(sem_raw.shape) == 4:
            sem_raw = np.squeeze(sem_raw)
        
        # Get single channel
        if len(sem_raw.shape) == 3:
            sem_img = sem_raw[..., 0]
        else:
            sem_img = sem_raw

        # Save Logic
        if env.engine.episode_step % SAVE_EVERY == 0:
            filename = f"f_{saved:06d}.jpg"
            # Save RGB
            cv2.imwrite(os.path.join(IMAGE_DIR, filename), rgb_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            
            # Store Mask in RAM
            mask_buffer.append(sem_img.astype(np.uint8))
            
            data_log.append([filename, env.agent.steering, env.agent.throttle_brake])
            saved += 1
            
            if saved % 100 == 0:
                print(f"📹 {saved} frames saved...")

        if terminated or truncated:
            env.reset()

except KeyboardInterrupt:
    print("\nStopping collection...")

finally:
    env.close()
    if data_log:
        pd.DataFrame(data_log, columns=["image", "steering", "throttle"]).to_csv(
            os.path.join(DATA_DIR, "driving_log.csv"), index=False
        )
        np.savez_compressed(os.path.join(DATA_DIR, "masks.npz"), masks=np.array(mask_buffer))
        print(f"✅ Success! {saved} frames saved to {DATA_DIR}/")
        print(f"   Images: {IMAGE_DIR}/")
        print(f"   Masks: {DATA_DIR}/masks.npz")
        print(f"   CSV: {DATA_DIR}/driving_log.csv")
#python record_data.py