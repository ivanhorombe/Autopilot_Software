import cv2
import cv2.aruco as aruco
import numpy as np
import glob

# --- SECTION 1: DEFINING THE "GROUND TRUTH" ---
# We are telling the computer exactly what the board looks like in the REAL world.
# If these numbers are wrong, the computer's sense of distance will be warped.
CHARUCO_BOARD_SHAPE = (7, 5)   # Grid dimensions: 7 squares across, 5 down.
SQUARE_LENGTH = 0.029          # The white/black chess squares are 29mm (0.029m).
MARKER_LENGTH = 0.0145         # The ArUco markers inside are 14.5mm (0.0145m).
# DICT_4X4_50 means the markers use a 4x4 bit grid and there are 50 possible unique IDs.
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# The 'board' object is a digital map of the perfect, flat, un-distorted board.
board = aruco.CharucoBoard(CHARUCO_BOARD_SHAPE, SQUARE_LENGTH, MARKER_LENGTH, ARUCO_DICT)
detector_params = aruco.DetectorParameters()
# The 'detector' is the logic engine that knows how to find black/white intersections.
detector = aruco.CharucoDetector(board)

# These lists will store the "matches" we find:
all_charuco_corners = [] # Where the corners are in the PIXELS (2D).
all_charuco_ids = []     # Which specific corner ID we found (so the math knows which is which).
image_size = None        # We need to know the resolution (e.g., 1080p) to scale the math.

# --- SECTION 2: IMAGE PROCESSING ---
# We loop through every photo you took to find as many "anchor points" as possible.
images = glob.glob('calibration_samples/*.png')
print(f"Found {len(images)} images. Processing...")

for fname in images:
    img = cv2.imread(fname)
    # Grayscale is faster and more accurate for finding edges (no color noise).
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if image_size is None:
        # Capture the width and height from the first image.
        image_size = gray.shape[::-1]

    # This is the "Magic" step. It finds the ArUco markers first, 
    # then uses them to find the high-precision chessboard corners.
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

    # We need at least 4 corners to solve the geometry of a 3D plane.
    if charuco_corners is not None and len(charuco_corners) > 4:
        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)
        print(f"✅ Found {len(charuco_corners)} corners in {fname}")
    else:
        # If the image is too blurry or at too extreme an angle, we toss it.
        print(f"❌ Could not find enough corners in {fname} - Skipping")


# --- SECTION 3: THE CORE MATHEMATICS ---
# This is where we compare the "Ideal 3D Board" to your "Real 2D Photos."
print("\nRunning calibration... (this may take a moment)")

all_obj_points = [] # Where the corners ARE in 3D space (0,0,0 ... 0.029,0,0).
all_img_points = [] # Where the corners APPEAR in your photo pixels.

for i in range(len(all_charuco_ids)):
    # board.getChessboardCorners() is the list of 3D coordinates for every corner.
    # We use the IDs we detected to pull ONLY the 3D points we actually saw in that photo.
    obj_pts = board.getChessboardCorners()[all_charuco_ids[i]]
    all_obj_points.append(obj_pts)
    all_img_points.append(all_charuco_corners[i])

# calibrateCamera is the "Solver." It tries to find a Camera Matrix and 
# Distortion Coefficients that make the 3D -> 2D projection match your photos.
# 'ret' (Reprojection Error) is the average pixel distance between where the 
# corners were found vs where the math says they SHOULD be.
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objectPoints=all_obj_points,
    imagePoints=all_img_points,
    imageSize=image_size,
    cameraMatrix=None,
    distCoeffs=None
)

# --- SECTION 4: OUTPUT & VALIDATION ---
if ret:
    # Save to a compressed file so other scripts (like the kart driver) can load it.
    np.savez('calibration_data.npz', mtx=camera_matrix, dist=dist_coeffs)
    print("\n✨ CALIBRATION SUCCESSFUL! ✨")
    print(f"Reprojection Error: {ret:.4f} pixels")
    
    # Focal Length: High numbers mean a "zoomed in" narrow view. 
    # Low numbers (like yours, ~1200) indicate a wide-angle lens.
    print("\nFocal Length (x, y):", camera_matrix[0,0], camera_matrix[1,1])
    
    # Principal Point: This is the optical center of the lens in pixels.
    print("Principal Point (cx, cy):", camera_matrix[0,2], camera_matrix[1,2])
else:
    print("\n💥 Calibration failed. Your photos might be too similar or too blurry.")
#python calibrate.py