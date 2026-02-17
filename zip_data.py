import shutil
import os
import time

# --- CONFIGURATION ---
DATA_PATH = "data"
IMAGE_PATH = os.path.join(DATA_PATH, "images")
SEED_FILE = "current_seed.txt"

# 1. IMPORT THE SEED FROM THE RECORDING SESSION
if os.path.exists(SEED_FILE):
    with open(SEED_FILE, "r") as f:
        session_seed = f.read().strip()
else:
    session_seed = "unknown"

# 2. VERIFICATION
print(f"Validating session data for Seed: {session_seed}...")

if os.path.exists(DATA_PATH) and os.path.exists(IMAGE_PATH):
    img_count = len(os.listdir(IMAGE_PATH))
    
    if img_count > 0:
        # 3. GENERATE FILENAME
        timestamp = time.strftime("%Y%m%d_%H%M") 
        unique_name = f"kart_data_{timestamp}_seed{session_seed}"

        try:
            print(f"Archiving: {unique_name}.zip")
            shutil.make_archive(base_name=unique_name, format="zip", root_dir=".", base_dir=DATA_PATH)
            
            if os.path.exists(f"{unique_name}.zip"):
                print(f"SUCCESS: {unique_name}.zip created.")
                
                # 4. CLEANUP EVERYTHING
                shutil.rmtree(DATA_PATH)
                if os.path.exists(SEED_FILE): os.remove(SEED_FILE) # Remove the bridge file
                print("Local data and seed file cleared.")
        except Exception as e:
            print(f"Error: {e}")
else:
    print("❌ No data found to zip.")