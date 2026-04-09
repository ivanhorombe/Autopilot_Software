import serial
import time

class KartHardware:
    """The Actuation Layer: Sends commands to ESP32 and reads feedback."""
    def __init__(self, port='COM4', baudrate=115200):
        try:
            # Connect with a short timeout so reading doesn't hang the loop
            self.ser = serial.Serial(port, baudrate, timeout=0.05)
            time.sleep(2) # ESP32 reboot delay
            print(f"--- SUCCESS: Connected to ESP32 on {port} ---")
        except Exception as e:
            print(f"--- ERROR: Could not connect to {port}. Check Port/Cable. ---")
            self.ser = None

    def send_command(self, steer, throttle):
        if self.ser and self.ser.is_open:
            # Map values
            steer_val = int((steer + 1) * 90)
            throttle_val = int(throttle * 255)

            # Send Packet
            packet = f"S{steer_val}T{throttle_val}\n"
            self.ser.write(packet.encode('utf-8'))

            # READ BACK: Listen for the ESP32's confirmation
            if self.ser.in_waiting > 0:
                try:
                    feedback = self.ser.readline().decode('utf-8').strip()
                    print(f"ESP32 CONFIRMATION: {feedback}")
                except:
                    pass # Ignore decoding glitches

    def close(self):
        if self.ser:
            self.ser.close()