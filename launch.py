"""
HAWKEYE CV-Engine Launcher
Simple launcher that works with the existing codebase
"""
import sys
import os

# Add cv_engine to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cv_engine'))

print("=" * 70)
print("  HAWKEYE CV-Engine // Phase 4 - Intelligence Orchestration")
print("=" * 70)
print()

# Check dependencies
print("Checking dependencies...")
missing = []

try:
    import cv2
    print("✓ OpenCV")
except ImportError:
    missing.append("opencv-python")
    print("✗ OpenCV")

try:
    from ultralytics import YOLO
    print("✓ Ultralytics YOLO")
except ImportError:
    missing.append("ultralytics")
    print("✗ Ultralytics")

try:
    import numpy as np
    print("✓ NumPy")
except ImportError:
    missing.append("numpy")
    print("✗ NumPy")

try:
    import flask
    print("✓ Flask")
except ImportError:
    missing.append("flask")
    print("✗ Flask")

try:
    import requests
    print("✓ Requests")
except ImportError:
    missing.append("requests")
    print("✗ Requests")

if missing:
    print(f"\n✗ Missing dependencies: {', '.join(missing)}")
    print(f"Install with: pip install {' '.join(missing)}")
    sys.exit(1)

print()
print("=" * 70)
print("Starting CV-Engine...")
print("=" * 70)
print()

# Import and run
try:
    # Use the original stream_processor
    os.chdir(os.path.dirname(__file__))
    exec(open('cv_engine/stream_processor.py').read())
except KeyboardInterrupt:
    print("\n\nShutting down...")
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
