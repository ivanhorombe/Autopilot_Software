import numpy as np

class MissionManager:
    """The Logic Layer: Decides behavior based on Marker IDs."""
    def __init__(self):
        self.latched_mode = "RACE_READY"
        self.max_throttle = 1.0
        self.steering_bias = 0.0

    def process_id(self, m_id, curr_z):
        """Updates internal state based on the ID seen."""
        if m_id == 5: # EMERGENCY STOP
            self.latched_mode = "E-STOP"
            self.max_throttle = 0.0
        
        elif m_id == 6: # SMOOTH STOP
            self.latched_mode = "SMOOTH_STOP"
            # Ramp down throttle based on distance (linear decay)
            self.max_throttle = np.clip((curr_z - 0.5) / 3.0, 0.0, 0.5)
            
        elif m_id == 7: # SLOW ZONE
            self.latched_mode = "SLOW_ZONE"
            self.max_throttle = 0.3
            
        elif m_id == 8: # SPEED UP
            self.latched_mode = "RACE_MODE"
            self.max_throttle = 1.0
            
        elif m_id == 9: # BIAS RIGHT
            self.latched_mode = "BIAS_RIGHT"
            self.steering_bias = 0.4
            
        elif m_id == 10: # BIAS LEFT
            self.latched_mode = "BIAS_LEFT"
            self.steering_bias = -0.4
            
        return self.latched_mode, self.max_throttle, self.steering_bias