"""
HAWKEYE CV-Engine - Minimal Working Launcher
Runs the CV engine with basic Phase 4 features
"""
import cv2
import time
import requests
import logging
import numpy as np
import threading
import queue
import os
import argparse
from flask import Flask, Response, request
from ultralytics import YOLO

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HAWKEYE")

# Flask app for video streaming
flask_app = Flask(__name__)
output_frame = None
lock = threading.Lock()

# Configuration from .env or defaults
BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:8000/api/v1/')
DEVICE = os.getenv('DEVICE', 'cpu')
VIDEO_SOURCE = os.getenv('VIDEO_SOURCE', 'people.mp4')
DETECTION_THRESHOLD = float(os.getenv('DETECTION_THRESHOLD', '0.4'))

print("=" * 70)
print("  HAWKEYE CV-Engine // Phase 4 - Intelligence Orchestration")
print("=" * 70)
print(f"  Device: {DEVICE}")
print(f"  Video Source: {VIDEO_SOURCE}")
print(f"  Backend: {BACKEND_URL}")
print(f"  Detection Threshold: {DETECTION_THRESHOLD}")
print("=" * 70)
print()

# Load YOLO model
logger.info(f"Loading YOLOv8 model on {DEVICE}...")
model = YOLO("yolov8n.pt")

if DEVICE in ['cuda', 'gpu']:
    try:
        model.to('cuda')
        model.half()
        logger.info("✓ YOLOv8 running on CUDA (FP16)")
    except:
        logger.warning("CUDA not available, falling back to CPU")
        model.to('cpu')
else:
    model.to('cpu')
    logger.info("✓ YOLOv8 running on CPU")

# Simple tracking
tracked_objects = {}
next_id = 1

def process_video():
    """Main video processing loop"""
    global output_frame, next_id
    
    logger.info(f"Opening video source: {VIDEO_SOURCE}")
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    if not cap.isOpened():
        logger.error(f"Failed to open video source: {VIDEO_SOURCE}")
        return
    
    fps_start = time.time()
    fps_counter = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            # Loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # Run YOLO detection
        results = model(frame, verbose=False, device=DEVICE)
        
        # Process detections
        for result in results:
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, cls = r
                cls = int(cls)
                
                # Only detect persons (0) and vehicles (2, 3, 5, 7)
                if cls in [0, 2, 3, 5, 7] and score > DETECTION_THRESHOLD:
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    
                    # Draw bounding box
                    color = (0, 255, 0) if cls == 0 else (255, 200, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Label
                    label = "PERSON" if cls == 0 else "VEHICLE"
                    cv2.putText(frame, f"{label} {score:.2f}", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # FPS counter
        fps_counter += 1
        if time.time() - fps_start > 1:
            logger.info(f"Performance: {fps_counter} FPS")
            fps_counter = 0
            fps_start = time.time()
        
        # Overlay
        cv2.putText(frame, "HAWKEYE v2.1 // PHASE 4 ACTIVE", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 65), 2)
        
        # Update output frame
        with lock:
            output_frame = frame.copy()

def generate():
    """Generate video frames for streaming"""
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encodedImage) + b'\r\n')

@flask_app.route("/video_feed")
def video_feed():
    """Video streaming route"""
    return Response(generate(), 
                   mimetype="multipart/x-mixed-replace; boundary=frame")

@flask_app.route("/")
def index():
    """Home page"""
    return """
    <html>
    <head><title>HAWKEYE CV-Engine</title></head>
    <body style="background: #000; color: #0f0; font-family: monospace;">
        <h1>🎯 HAWKEYE CV-Engine // Phase 4</h1>
        <h2>Live Video Feed</h2>
        <img src="/video_feed" width="100%">
        <p>Status: ONLINE | Device: """ + DEVICE + """ | FPS: Real-time</p>
    </body>
    </html>
    """

if __name__ == "__main__":
    # Start video processing thread
    video_thread = threading.Thread(target=process_video, daemon=True)
    video_thread.start()
    
    logger.info("✓ Video processing started")
    logger.info("✓ Flask server starting on http://0.0.0.0:5001")
    logger.info("✓ Access video feed at: http://localhost:5001")
    print()
    print("=" * 70)
    print("  🚀 HAWKEYE CV-Engine is RUNNING")
    print("  📹 Video Feed: http://localhost:5001")
    print("  ⚡ Press Ctrl+C to stop")
    print("=" * 70)
    print()
    
    # Start Flask server
    flask_app.run(host="0.0.0.0", port=5001, debug=False, threaded=True, use_reloader=False)
