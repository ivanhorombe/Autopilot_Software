import gymnasium as gym
from metadrive.envs.metadrive_env import MetaDriveEnv

def main():
    # Only using verified keys from your terminal logs to ensure zero crashes
    config = {
    "use_render": True,
    "manual_control": True,
    "map": 5,
    "traffic_density": 0.1,
    "image_observation": True,  # Keep this True
    "window_size": (800, 600),
    "show_sidewalk": True,
    "show_crosswalk": True,
    "show_interface": True,
    
    "vehicle_config": {
        # CRITICAL FIX: Match 'image_source' to one of the existing keys in your error
        "image_source": "main_camera", 
        "width": 1.2,
        "length": 1.8,
        "mass": 200,
    },
}
    env = MetaDriveEnv(config)
    
    try:
        obs = env.reset()
        print("Simulator Started. CLICK the window and press 'W' to drive.")
        print("If frozen, press 'P' to unpause.")
        
        while True:
            # env.step([0, 0]) is the heartbeat of the physics engine.
            # In manual_control mode, [0, 0] is overwritten by your keyboard.
            obs, reward, terminated, truncated, info = env.step([0, 0])
            
            # The render call updates the visual window
            env.render()
            
            if terminated or truncated:
                print("Episode ended (Crash or Out of Road). Resetting...")
                env.reset()
                
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        env.close()

if __name__ == "__main__":
    main()
#python simulation/run_sim.py