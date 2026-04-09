import numpy as np

class PIDController:
    """Handles the math for steering logic (Sense -> Logic)."""
    def __init__(self, kp, kd):
        self.kp = kp
        self.kd = kd
        self.prev_error = 0
        
    def compute(self, error, dt):
        """Calculates the steering output based on error and time delta."""
        if dt <= 0: return 0
        
        p_term = self.kp * error
        d_term = self.kd * (error - self.prev_error) / dt
        
        self.prev_error = error
        return np.clip(p_term + d_term, -1.0, 1.0)

class EMAFilter:
    """Handles Exponential Moving Average to smooth noisy sensor data."""
    def __init__(self, alpha, initial_value=0.0):
        self.alpha = alpha # Smoothing factor (0.0 to 1.0)
        self.value = initial_value
        
    def apply(self, new_value):
        """Filters a new reading and returns the smoothed result."""
        self.value = (new_value * self.alpha) + (self.value * (1 - self.alpha))
        return self.value
    
class Odometer:
    """The Physics Engine: Calculates speed based on distance changes."""
    def __init__(self, alpha=0.2):
        self.filter = EMAFilter(alpha)
        self.prev_z = None
        self.current_velocity = 0.0

    def update(self, curr_z, dt):
        """Calculates smoothed velocity. Returns meters per second."""
        if self.prev_z is not None and dt > 0:
            # Physics: velocity = delta_distance / delta_time
            inst_velocity = (self.prev_z - curr_z) / dt
            self.current_velocity = self.filter.apply(inst_velocity)
        
        self.prev_z = curr_z
        return self.current_velocity