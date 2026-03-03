"""
Simple startup script for HAWKEYE CV-Engine
Runs with basic dependencies to test the system
"""
import sys
import os

# Add cv_engine to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cv_engine'))

print("=" * 60)
print("HAWKEYE CV-Engine - Simple Startup")
print("=" * 60)

# Check Python version
print(f"Python Version: {sys.version}")

# Check critical imports
print("\nChecking dependencies...")
try:
    import cv2
    print("✓ OpenCV installed")
except ImportError:
    print("✗ OpenCV not installed - run: pip install opencv-python")
    sys.exit(1)

try:
    import numpy as np
    print("✓ NumPy installed")
except ImportError:
    print("✗ NumPy not installed - run: pip install numpy")
    sys.exit(1)

try:
    import flask
    print("✓ Flask installed")
except ImportError:
    print("✗ Flask not installed - run: pip install flask")
    sys.exit(1)

try:
    import requests
    print("✓ Requests installed")
except ImportError:
    print("✗ Requests not installed - run: pip install requests")
    sys.exit(1)

# Check for video file
video_file = "people.mp4"
if not os.path.exists(video_file):
    print(f"\n✗ Video file not found: {video_file}")
    print("Please ensure people.mp4 is in the cv-engine directory")
    sys.exit(1)
else:
    print(f"✓ Video file found: {video_file}")

# Check for YOLO model
model_file = "yolov8n.pt"
if not os.path.exists(model_file):
    print(f"\n⚠ YOLO model not found: {model_file}")
    print("The model will be downloaded automatically on first run")
else:
    print(f"✓ YOLO model found: {model_file}")

print("\n" + "=" * 60)
print("Starting CV-Engine with basic configuration...")
print("=" * 60 + "\n")

# Import and run the original stream processor
try:
    from stream_processor import StreamProcessor
    
    processor = StreamProcessor(
        source=video_file,
        backend_url="http://localhost:8000/api/v1/",
        detection_threshold=0.4,
        device="cpu"
    )
    
    print("✓ CV-Engine initialized successfully")
    print("\nStarting video processing...")
    print("Access video feed at: http://localhost:5001/video_feed")
    print("Press Ctrl+C to stop\n")
    
    import threading
    t = threading.Thread(target=processor.process_stream)
    t.daemon = True
    t.start()
    
    # Start Flask server
    from flask import Flask
    flask_app = Flask(__name__)
    flask_app.run(host="0.0.0.0", port=5001, debug=False, threaded=True, use_reloader=False)
    
except KeyboardInterrupt:
    print("\n\nShutting down CV-Engine...")
    processor.is_running = False
    print("✓ Shutdown complete")
except Exception as e:
    print(f"\n✗ Error starting CV-Engine: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
