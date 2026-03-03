"""
HAWKEYE CV-Engine - Complete System with Mini Backend
Full face recognition, forensic search, and breadcrumb trails
"""
import cv2
import time
import logging
import numpy as np
import threading
import os
from flask import Flask, Response, request, jsonify, render_template_string
from ultralytics import YOLO

# Import mini backend
import sys
sys.path.insert(0, os.path.dirname(__file__))
from mini_backend.simple_face_db import SimpleFaceDB
from mini_backend.sighting_tracker import SightingTracker

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HAWKEYE_COMPLETE")

# Flask app
flask_app = Flask(__name__)
output_frame = None
lock = threading.Lock()

# Configuration
DEVICE = os.getenv('DEVICE', 'cpu')
VIDEO_SOURCE = os.getenv('VIDEO_SOURCE', 'people.mp4')
DETECTION_THRESHOLD = float(os.getenv('DETECTION_THRESHOLD', '0.4'))

print("=" * 70)
print("  HAWKEYE CV-Engine // COMPLETE SYSTEM")
print("  Face Recognition + Forensic Search + Breadcrumb Trails")
print("=" * 70)
print(f"  Device: {DEVICE}")
print(f"  Video Source: {VIDEO_SOURCE}")
print(f"  Face Recognition: ENABLED (OpenCV)")
print("=" * 70)
print()

# Initialize components
logger.info("Initializing YOLO...")
model = YOLO("yolov8n.pt")
model.to(DEVICE)

logger.info("Initializing Face Database...")
face_db = SimpleFaceDB()

logger.info("Initializing Sighting Tracker...")
tracker = SightingTracker()

# Tracking state
registered_first_person = False
last_sighting_time = {}  # {person_name: timestamp}
SIGHTING_INTERVAL = 2  # seconds between sightings

def process_video():
    """Main video processing loop"""
    global output_frame, registered_first_person
    
    logger.info(f"Opening video source: {VIDEO_SOURCE}")
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    if not cap.isOpened():
        logger.error(f"Failed to open video source")
        return
    
    fps_start = time.time()
    fps_counter = 0
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            registered_first_person = False
            continue
        
        frame_count += 1
        
        # Run YOLO detection
        results = model(frame, verbose=False, device=DEVICE)
        
        # Process detections
        for result in results:
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, cls = r
                cls = int(cls)
                
                if cls in [0, 2, 3, 5, 7] and score > DETECTION_THRESHOLD:
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    
                    # PERSON DETECTION with Face Recognition
                    if cls == 0:
                        # Extract person region
                        person_roi = frame[max(0, y1):min(frame.shape[0], y2), 
                                         max(0, x1):min(frame.shape[1], x2)]
                        
                        if person_roi.size == 0:
                            continue
                        
                        # Register first person as "Juma Macharia"
                        if not registered_first_person:
                            if face_db.register_face(person_roi, "Juma Macharia"):
                                registered_first_person = True
                                logger.info("🎯 TARGET REGISTERED: Juma Macharia")
                        
                        # Identify person
                        name, conf, face_crop = face_db.identify(person_roi)
                        
                        # Record sighting (throttled)
                        current_time = time.time()
                        if name != "Unknown":
                            last_time = last_sighting_time.get(name, 0)
                            if current_time - last_time > SIGHTING_INTERVAL:
                                tracker.add_sighting(
                                    person_name=name,
                                    confidence=conf,
                                    camera_id="CAM-LIVE-01",
                                    location="Grid Sector Alpha",
                                    bbox=(x1, y1, x2-x1, y2-y1)
                                )
                                last_sighting_time[name] = current_time
                        
                        # Visual styling based on identification
                        if name != "Unknown":
                            # RED for known targets
                            color = (0, 0, 255)
                            label = f"🎯 TARGET: {name}"
                            thickness = 3
                            
                            # Draw filled background for label
                            (label_w, label_h), _ = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                            )
                            cv2.rectangle(frame, (x1, y1-label_h-15), 
                                        (x1+label_w+10, y1), color, -1)
                            cv2.putText(frame, label, (x1+5, y1-8),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            
                            # Confidence bar
                            bar_width = int((x2 - x1) * conf)
                            cv2.rectangle(frame, (x1, y2+5), 
                                        (x1+bar_width, y2+15), (0, 0, 255), -1)
                            cv2.putText(frame, f"{conf:.0%}", (x1, y2+30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        else:
                            # GREEN for unknown persons
                            color = (0, 255, 65)
                            label = f"PERSON ({score:.2f})"
                            thickness = 2
                            cv2.putText(frame, label, (x1, y1-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # VEHICLE DETECTION
                    elif cls in [2, 3, 5, 7]:
                        color = (255, 200, 0)
                        veh_types = {2: 'CAR', 3: 'MOTO', 5: 'BUS', 7: 'TRUCK'}
                        label = f"{veh_types.get(cls, 'VEHICLE')} ({score:.2f})"
                        
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
        cv2.putText(frame, "HAWKEYE COMPLETE // FACE RECOGNITION ACTIVE", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 65), 2)
        
        # Show registered targets
        known_faces = face_db.get_all_faces()
        if known_faces:
            targets_text = f"TARGETS: {', '.join(known_faces)}"
            cv2.putText(frame, targets_text, (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Show sighting count
        stats = tracker.get_statistics()
        stats_text = f"SIGHTINGS: {stats['total_sightings']}"
        cv2.putText(frame, stats_text, (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Update output frame
        with lock:
            output_frame = frame.copy()

def generate():
    """Generate video frames"""
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
    return Response(generate(), 
                   mimetype="multipart/x-mixed-replace; boundary=frame")

@flask_app.route("/")
def index():
    """Dashboard with forensic search"""
    known_faces = face_db.get_all_faces()
    stats = tracker.get_statistics()
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>HAWKEYE Complete System</title>
        <style>
            body {
                background: #000;
                color: #0f0;
                font-family: 'Courier New', monospace;
                margin: 0;
                padding: 20px;
            }
            h1 { color: #0f0; border-bottom: 2px solid #0f0; padding-bottom: 10px; }
            h2 { color: #ff0; }
            .container { max-width: 1400px; margin: 0 auto; }
            .video-container { margin: 20px 0; }
            .video-container img { width: 100%; border: 2px solid #0f0; }
            .panel {
                background: #111;
                border: 1px solid #0f0;
                padding: 15px;
                margin: 10px 0;
            }
            .target { color: #f00; font-weight: bold; }
            .stat { color: #ff0; }
            button {
                background: #0f0;
                color: #000;
                border: none;
                padding: 10px 20px;
                font-family: 'Courier New', monospace;
                font-weight: bold;
                cursor: pointer;
                margin: 5px;
            }
            button:hover { background: #0ff; }
            .breadcrumb { background: #222; padding: 10px; margin: 5px 0; border-left: 3px solid #f00; }
            .sighting { background: #1a1a1a; padding: 8px; margin: 3px 0; border-left: 2px solid #0f0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 HAWKEYE COMPLETE SYSTEM</h1>
            <p>Face Recognition + Forensic Search + Breadcrumb Trails</p>
            
            <div class="video-container">
                <h2>📹 Live Video Feed</h2>
                <img src="/video_feed" />
            </div>
            
            <div class="panel">
                <h2>🎯 Registered Targets</h2>
                <p class="target">{{ known_faces|join(', ') or 'None' }}</p>
                <p class="stat">Total Sightings: {{ stats['total_sightings'] }}</p>
            </div>
            
            <div class="panel">
                <h2>🔍 Forensic Search</h2>
                {% for person in known_faces %}
                <button onclick="searchPerson('{{ person }}')">Search: {{ person }}</button>
                <button onclick="getBreadcrumb('{{ person }}')">Breadcrumb Trail: {{ person }}</button>
                {% endfor %}
                <button onclick="getRecent()">Recent Sightings</button>
            </div>
            
            <div class="panel" id="results">
                <h2>📊 Results</h2>
                <p>Click a button above to search...</p>
            </div>
        </div>
        
        <script>
            function searchPerson(name) {
                fetch('/api/forensic_search?person=' + name)
                    .then(r => r.json())
                    .then(data => {
                        let html = '<h2>🔍 Forensic Search: ' + name + '</h2>';
                        html += '<p>Found ' + data.results.length + ' sightings</p>';
                        data.results.forEach(s => {
                            html += '<div class="sighting">';
                            html += '<strong>' + s.timestamp + '</strong> - ';
                            html += s.camera_id + ' (' + s.location + ') - ';
                            html += 'Confidence: ' + (s.confidence * 100).toFixed(0) + '%';
                            html += '</div>';
                        });
                        document.getElementById('results').innerHTML = html;
                    });
            }
            
            function getBreadcrumb(name) {
                fetch('/api/breadcrumb?person=' + name)
                    .then(r => r.json())
                    .then(data => {
                        let html = '<h2>🗺️ Breadcrumb Trail: ' + name + '</h2>';
                        html += '<p>' + data.trail.length + ' locations tracked</p>';
                        data.trail.forEach((s, i) => {
                            html += '<div class="breadcrumb">';
                            html += '<strong>Step ' + (i+1) + ':</strong> ' + s.timestamp + ' - ';
                            html += s.camera_id + ' (' + s.location + ')';
                            html += '</div>';
                        });
                        document.getElementById('results').innerHTML = html;
                    });
            }
            
            function getRecent() {
                fetch('/api/recent_sightings')
                    .then(r => r.json())
                    .then(data => {
                        let html = '<h2>📊 Recent Sightings</h2>';
                        data.sightings.forEach(s => {
                            html += '<div class="sighting">';
                            html += '<strong>' + s.person_name + '</strong> - ';
                            html += s.timestamp + ' - ' + s.camera_id;
                            html += '</div>';
                        });
                        document.getElementById('results').innerHTML = html;
                    });
            }
        </script>
    </body>
    </html>
    """
    
    from flask import render_template_string
    return render_template_string(html, 
                                  known_faces=known_faces, 
                                  stats=stats)

@flask_app.route("/api/forensic_search")
def api_forensic_search():
    """Forensic search API"""
    person = request.args.get('person')
    camera = request.args.get('camera')
    
    results = tracker.forensic_search(person_name=person, camera_id=camera)
    return jsonify({'results': results})

@flask_app.route("/api/breadcrumb")
def api_breadcrumb():
    """Breadcrumb trail API"""
    person = request.args.get('person')
    trail = tracker.get_breadcrumb_trail(person)
    return jsonify({'trail': trail})

@flask_app.route("/api/recent_sightings")
def api_recent_sightings():
    """Recent sightings API"""
    sightings = tracker.get_recent_sightings(limit=50)
    return jsonify({'sightings': sightings})

@flask_app.route("/api/statistics")
def api_statistics():
    """Statistics API"""
    stats = tracker.get_statistics()
    return jsonify(stats)

if __name__ == "__main__":
    # Start video processing
    video_thread = threading.Thread(target=process_video, daemon=True)
    video_thread.start()
    
    logger.info("✓ Video processing started")
    logger.info("✓ Flask server starting on http://0.0.0.0:5001")
    print()
    print("=" * 70)
    print("  🚀 HAWKEYE COMPLETE SYSTEM RUNNING")
    print("  📹 Dashboard: http://localhost:5001")
    print("  🎯 Face Recognition: ENABLED")
    print("  🔍 Forensic Search: ENABLED")
    print("  🗺️ Breadcrumb Trails: ENABLED")
    print("  ⚡ Press Ctrl+C to stop")
    print("=" * 70)
    print()
    
    flask_app.run(host="0.0.0.0", port=5001, debug=False, threaded=True, use_reloader=False)
