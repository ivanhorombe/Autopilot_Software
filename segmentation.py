import pandas as pd
import cv2
import os
import numpy as np

# --- CONFIGURATION ---
DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "driving_log.csv")

def play_back_session():
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: {CSV_PATH} not found.")
        return

    # Load the log
    df = pd.read_csv(CSV_PATH)
    print(f"🎬 Playing back {len(df)} frames. Press 'q' to exit.")

    for index, row in df.iterrows():
        # Load the two images
        rgb_path = os.path.join(DATA_DIR, row['image'])
        mask_path = os.path.join(DATA_DIR, row['mask'])

        rgb_img = cv2.imread(rgb_path)
        mask_img = cv2.imread(mask_path)

        if rgb_img is None or mask_img is None:
            continue

        # Colorize the mask so it's easier to see (Pseudo-coloring)
        # Raw masks are often very dark because class values are small (1, 2, 3...)
        color_mask = cv2.applyColorMap(mask_img, cv2.COLORMAP_JET)

        # Stitch them together side-by-side
        combined_view = np.hstack((rgb_img, color_mask))

        # Add an overlay for steering/throttle
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined_view, f"Steer: {row['steering']:.2f}", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(combined_view, f"Thr: {row['throttle']:.2f}", (10, 60), font, 0.7, (255, 255, 255), 2)
        cv2.putText(combined_view, "RAW RGB VIEW | SEMANTIC MASK", (300, 290), font, 0.6, (255, 255, 255), 1)

        cv2.imshow("Data Integrity Check", combined_view)

        # Control playback speed (33ms = ~30fps)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    play_back_session()