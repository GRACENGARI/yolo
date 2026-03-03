"""
HAWKEYE CV-Engine - Full Version with Face Recognition
Includes ArcFace face identification and target person tracking
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

# Try to import InsightFace for face recognition
try:
    import insightface
    from insightface.app import FaceAnalysis
    HAS_INSIGHTFACE = True
    logger.info("✓ InsightFace available - Face recognition ENABLED")
except ImportError:
    HAS_INSIGHTFACE = False
    logger.warning("⚠ InsightFace not installed - Face recognition DISABLED")
    logger.info("Install with: pip install insightface onnxruntime")

# Flask app for video streaming
flask_app = Flask(__name__)
output_frame = None
lock = threading.Lock()

# Configuration
BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:8000/api/v1/')
DEVICE = os.getenv('DEVICE', 'cpu')
VIDEO_SOURCE = os.getenv('VIDEO_SOURCE', 'people.mp4')
DETECTION_THRESHOLD = float(os.getenv('DETECTION_THRESHOLD', '0.4'))

print("=" * 70)
print("  HAWKEYE CV-Engine // Phase 4 - Full Intelligence System")
print("=" * 70)
print(f"  Device: {DEVICE}")
print(f"  Video Source: {VIDEO_SOURCE}")
print(f"  Backend: {BACKEND_URL}")
print(f"  Face Recognition: {'ENABLED' if HAS_INSIGHTFACE else 'DISABLED'}")
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

# Face Recognition Setup
class FaceIdentifier:
    def __init__(self):
        self.known_faces = {}  # {name: embedding}
        self.enabled = HAS_INSIGHTFACE
        
        if self.enabled:
            try:
                providers = ['CPUExecutionProvider']
                self.app = FaceAnalysis(name='buffalo_s', providers=providers)
                self.app.prepare(ctx_id=0, det_size=(640, 640))
                logger.info("✓ ArcFace (buffalo_s) initialized")
            except Exception as e:
                logger.error(f"Failed to load ArcFace: {e}")
                self.enabled = False
    
    def register_face(self, image, name):
        """Register a face with a name"""
        if not self.enabled:
            return False
        
        try:
            faces = self.app.get(image)
            if len(faces) == 0:
                return False
            
            # Take the largest face
            faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
            embedding = faces[0].embedding
            self.known_faces[name] = embedding
            logger.info(f"✓ Registered: {name}")
            return True
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False
    
    def identify(self, face_crop, threshold=0.4):
        """Identify a face"""
        if not self.enabled:
            return "Unknown", 0.0, None
        
        try:
            faces = self.app.get(face_crop)
            if len(faces) == 0:
                return "Unknown", 0.0, None
            
            target_embedding = faces[0].embedding
            
            best_score = -1.0
            best_name = "Unknown"
            
            for name, known_embedding in self.known_faces.items():
                # Cosine similarity
                score = np.dot(target_embedding, known_embedding) / (
                    np.linalg.norm(target_embedding) * np.linalg.norm(known_embedding)
                )
                if score > best_score:
                    best_score = score
                    best_name = name
            
            if best_score > threshold:
                return best_name, float(best_score), target_embedding
            
            return "Unknown", float(best_score), target_embedding
        except Exception as e:
            logger.error(f"Identification error: {e}")
            return "Unknown", 0.0, None

# Initialize face identifier
face_id = FaceIdentifier()

# Tracking state
tracked_objects = {}
next_id = 1
registered_first_person = False

def process_video():
    """Main video processing loop with face recognition"""
    global output_frame, next_id, registered_first_person
    
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
            registered_first_person = False  # Reset for demo
            continue
        
        # Run YOLO detection
        results = model(frame, verbose=False, device=DEVICE)
        
        # Process detections
        for result in results:
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, cls = r
                cls = int(cls)
                
                # Detect persons (0) and vehicles (2, 3, 5, 7)
                if cls in [0, 2, 3, 5, 7] and score > DETECTION_THRESHOLD:
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    
                    # PERSON DETECTION with Face Recognition
                    if cls == 0 and face_id.enabled:
                        # Extract face region
                        face_roi = frame[max(0, y1):min(frame.shape[0], y2), 
                                       max(0, x1):min(frame.shape[1], x2)]
                        
                        # Register first person as "Juma Macharia" (demo)
                        if not registered_first_person and face_roi.size > 0:
                            if face_id.register_face(face_roi, "Juma Macharia"):
                                registered_first_person = True
                                logger.info("🎯 TARGET REGISTERED: Juma Macharia")
                        
                        # Try to identify
                        name, conf, embedding = face_id.identify(face_roi)
                        
                        # Color based on identification
                        if name != "Unknown":
                            color = (0, 0, 255)  # RED for known targets
                            label = f"TARGET: {name} ({conf:.2f})"
                            thickness = 3
                        else:
                            color = (0, 255, 65)  # Green for unknown persons
                            label = f"PERSON ({score:.2f})"
                            thickness = 2
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                        
                        # Draw label with background
                        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(frame, (x1, y1-label_h-10), (x1+label_w, y1), color, -1)
                        cv2.putText(frame, label, (x1, y1-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Add confidence bar
                        if name != "Unknown":
                            bar_width = int((x2 - x1) * conf)
                            cv2.rectangle(frame, (x1, y2+5), (x1+bar_width, y2+15), (0, 0, 255), -1)
                    
                    # VEHICLE DETECTION
                    elif cls in [2, 3, 5, 7]:
                        color = (255, 200, 0)  # Cyan/Orange for vehicles
                        veh_types = {2: 'CAR', 3: 'MOTO', 5: 'BUS', 7: 'TRUCK'}
                        label = f"{veh_types.get(cls, 'VEHICLE')} ({score:.2f})"
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # PERSON without face recognition
                    elif cls == 0:
                        color = (0, 255, 65)
                        label = f"PERSON ({score:.2f})"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # FPS counter
        fps_counter += 1
        if time.time() - fps_start > 1:
            logger.info(f"Performance: {fps_counter} FPS")
            fps_counter = 0
            fps_start = time.time()
        
        # Overlay
        status_text = "HAWKEYE v2.1 // PHASE 4 - FACE RECOGNITION ACTIVE"
        cv2.putText(frame, status_text, (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 65), 2)
        
        # Show registered targets
        if len(face_id.known_faces) > 0:
            targets_text = f"TARGETS: {', '.join(face_id.known_faces.keys())}"
            cv2.putText(frame, targets_text, (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
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
    targets = ', '.join(face_id.known_faces.keys()) if face_id.known_faces else 'None'
    return f"""
    <html>
    <head><title>HAWKEYE CV-Engine - Full System</title></head>
    <body style="background: #000; color: #0f0; font-family: monospace;">
        <h1>🎯 HAWKEYE CV-Engine // Phase 4 - Full Intelligence</h1>
        <h2>Live Video Feed with Face Recognition</h2>
        <img src="/video_feed" width="100%">
        <p>Status: ONLINE | Device: {DEVICE} | Face Recognition: {'ENABLED' if face_id.enabled else 'DISABLED'}</p>
        <p>Registered Targets: {targets}</p>
        <p style="color: #ff0000;">🔴 RED boxes = Known targets | 🟢 GREEN boxes = Unknown persons | 🟠 ORANGE boxes = Vehicles</p>
    </body>
    </html>
    """

@flask_app.route("/register", methods=['POST'])
def register_target():
    """API endpoint to register a new target"""
    # This would be used by the backend to register new targets
    return {"status": "Feature available in full backend integration"}, 200

if __name__ == "__main__":
    # Start video processing thread
    video_thread = threading.Thread(target=process_video, daemon=True)
    video_thread.start()
    
    logger.info("✓ Video processing started")
    logger.info("✓ Flask server starting on http://0.0.0.0:5001")
    logger.info("✓ Access video feed at: http://localhost:5001")
    print()
    print("=" * 70)
    print("  🚀 HAWKEYE CV-Engine FULL SYSTEM is RUNNING")
    print("  📹 Video Feed: http://localhost:5001")
    print("  🎯 Face Recognition: " + ("ENABLED" if face_id.enabled else "DISABLED"))
    print("  ⚡ Press Ctrl+C to stop")
    print("=" * 70)
    print()
    
    # Start Flask server
    flask_app.run(host="0.0.0.0", port=5001, debug=False, threaded=True, use_reloader=False)
