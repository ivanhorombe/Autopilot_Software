import cv2

class Dashboard:
    """The Visualization Layer: Renders telemetry and status to the screen."""
    def __init__(self, window_name="Formula E AI Master Controller"):
        self.window_name = window_name

    def draw(self, frame, velocity, steer, throttle, mode, status_msg, status_color, dist=None):
        # Draw Background Overlay
        cv2.rectangle(frame, (0, 0), (650, 180), (0, 0, 0), -1)
        
        # 1. Speedometer
        speed_mph = abs(velocity) * 2.237
        cv2.putText(frame, f"SPD: {abs(velocity):.2f} m/s ({speed_mph:.1f} mph)", (20, 40), 1, 1.5, (255, 255, 255), 2)
        
        # 2. Steering Bar
        bar_x = int(325 + (steer * 200)) # Center + offset
        cv2.line(frame, (125, 80), (525, 80), (100, 100, 100), 2)
        cv2.circle(frame, (bar_x, 80), 12, (0, 255, 255), -1)
        cv2.putText(frame, "STEER", (20, 85), 1, 1.0, (255, 255, 255), 1)

        # 3. Throttle, Mode, and Distance
        dist_text = f"DIST: {dist:.2f}m" if dist is not None else "DIST: ---"
        cv2.putText(frame, f"THR: {throttle*100:.0f}%  | {dist_text}", (20, 120), 1, 1.2, (255, 255, 255), 1)
        cv2.putText(frame, f"{mode}: {status_msg}", (20, 160), 1, 1.2, status_color, 2)

        cv2.imshow(self.window_name, frame)