import cv2
import cv2.aruco as aruco
import numpy as np

class ArUcoTracker:
    """The Perception Layer: Translates pixels into 3D real-world coordinates."""
    
    def __init__(self, calibration_file, marker_size):
        # Load the 'eyes' of the kart
        with np.load(calibration_file) as data:
            self.cam_matrix = data['mtx']
            self.dist_coeffs = data['dist']
        
        self.marker_size = marker_size
        
        # Initialize ArUco detector
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.detector = aruco.ArucoDetector(self.aruco_dict, aruco.DetectorParameters())
        
        # Define 3D points of a flat marker (the 'Object Points')
        # This tells SolvePnP what a square looks like in the real world.
        self.obj_pts = np.array([
            [-marker_size/2,  marker_size/2, 0], 
            [ marker_size/2,  marker_size/2, 0], 
            [ marker_size/2, -marker_size/2, 0], 
            [-marker_size/2, -marker_size/2, 0]
        ], dtype=np.float32)

    def get_target(self, frame):
        """
        Processes a frame and returns (ID, X, Z) if a marker is found.
        X = Horizontal offset (m), Z = Distance (m)
        """
        corners, ids, _ = self.detector.detectMarkers(frame)
        
        if ids is not None:
            # We track the first marker detected
            idx = 0
            m_id = ids[idx][0]
            
            # Solve Perspective-n-Point (The 3D Math)
            _, _, tvec = cv2.solvePnP(self.obj_pts, corners[idx], self.cam_matrix, self.dist_coeffs)
            
            # tvec[0] is X (left/right), tvec[2] is Z (forward)
            curr_x = tvec[0][0]
            curr_z = tvec[2][0]
            
            return m_id, curr_x, curr_z, tvec, corners
            
        return None, None, None, None, None