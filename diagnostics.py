import sys
import os

def check_setup():
    print("--- 🏁 Kart Project Diagnostic 🏁 ---")
    
    # 1. Check Python Version
    print(f"Python Version: {sys.version.split()[0]}")
    if sys.version_info.major == 3 and sys.version_info.minor in [9, 10]:
        print("✅ Python version is optimal (3.9/3.10).")
    else:
        print("⚠️ Warning: Non-standard Python version detected.")

    # 2. Check OpenCV
    try:
        import cv2
        print(f"✅ OpenCV version {cv2.__version__} installed.")
    except ImportError:
        print("❌ OpenCV NOT FOUND. Run: pip install opencv-python")

    # 3. Check MetaDrive
    try:
        from metadrive.envs.metadrive_env import MetaDriveEnv
        print("✅ MetaDrive Simulator installed.")
    except ImportError:
        print("❌ MetaDrive NOT FOUND. Run: pip install metadrive-simulator")

    # 4. Check Serial (ESP32 Bridge)
    try:
        import serial
        import serial.tools.list_ports
        print("✅ PySerial installed.")
        ports = list(serial.tools.list_ports.comports())
        print(f"ℹ️ Found {len(ports)} active serial ports.")
    except ImportError:
        print("❌ PySerial NOT FOUND. Run: pip install pyserial")

if __name__ == "__main__":
    check_setup()