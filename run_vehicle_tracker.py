"""
HAWKEYE Vehicle Tracker - Vehicle-Only Surveillance System
Professional vehicle tracking with target registration
"""
import cv2
import time
import logging
import numpy as np
import threading
import os
from flask import Flask, Response, request, jsonify, render_template_string
from ultralytics import YOLO
import sys
sys.path.insert(0, os.path.dirname(__file__))
from mini_backend.sighting_tracker import SightingTracker
from pathlib import Path

# Load environment variables
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VEHICLE_TRACKER")

# Flask app
flask_app = Flask(__name__)
output_frame = None
lock = threading.Lock()

# Configuration
DEVICE = os.getenv('DEVICE', 'cpu')
VIDEO_SOURCE = os.getenv('VIDEO_SOURCE', 'surveillance_video.mp4')
DETECTION_THRESHOLD = 0.5  # Lower threshold for better vehicle detection

print("=" * 70)
print("  HAWKEYE VEHICLE TRACKER")
print("  Professional Vehicle Surveillance System")
print("=" * 70)
print(f"  Device: {DEVICE}")
print(f"  Video Source: {VIDEO_SOURCE}")
print(f"  Detection Threshold: {DETECTION_THRESHOLD}")
print("=" * 70)
print()

# Initialize
logger.info("Initializing YOLO...")
model = YOLO("yolov8n.pt")
model.to(DEVICE)

logger.info("Initializing Sighting Tracker...")
tracker = SightingTracker()

# Tracking state
vehicle_tracks = {}  # {track_id: {'description': str, 'is_target': bool, ...}}
next_vehicle_id = 1
target_vehicles = {}  # {track_id: description}
last_sighting_time = {}

def get_vehicle_info(track_id):
    """Get or create vehicle info for track ID"""
    global next_vehicle_id
    
    if track_id not in vehicle_tracks:
        is_target = track_id in target_vehicles
        description = target_vehicles.get(track_id, f"Vehicle-{next_vehicle_id}")
        
        if not is_target:
            next_vehicle_id += 1
        
        vehicle_tracks[track_id] = {
            'description': description,
            'is_target': is_target,
            'first_seen': time.time(),
            'last_seen': time.time(),
            'sightings': 0
        }
        
        if is_target:
            logger.info(f"🚗 TARGET VEHICLE: {description} (Track ID: {track_id})")
    
    vehicle_tracks[track_id]['last_seen'] = time.time()
    vehicle_tracks[track_id]['sightings'] += 1
    
    return vehicle_tracks[track_id]

def register_vehicle_target(track_id, description):
    """Register a vehicle as a target"""
    target_vehicles[track_id] = description
    if track_id in vehicle_tracks:
        vehicle_tracks[track_id]['description'] = description
        vehicle_tracks[track_id]['is_target'] = True
    logger.info(f"✓ Registered vehicle target: {description} (Track ID: {track_id})")

def process_video():
    """Main video processing loop"""
    global output_frame
    
    logger.info(f"Opening video source: {VIDEO_SOURCE}")
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    if not cap.isOpened():
        logger.error("Failed to open video source")
        return
    
    fps_start = time.time()
    fps_counter = 0
    
    # Track history for trails
    from collections import defaultdict
    track_history = defaultdict(lambda: [])
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # Run YOLO with tracking
        results = model.track(frame, persist=True, verbose=False, device=DEVICE, classes=[2, 3, 5, 7])  # Only vehicles
        
        # Process detections
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, track_id, conf, cls in zip(boxes, track_ids, confidences, classes):
                if conf > DETECTION_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Get vehicle info
                    vehicle_info = get_vehicle_info(track_id)
                    vehicle_desc = vehicle_info['description']
                    is_target = vehicle_info['is_target']
                    
                    # Record vehicle sighting (throttled)
                    current_time = time.time()
                    last_time = last_sighting_time.get(vehicle_desc, 0)
                    if current_time - last_time > 2:
                        tracker.add_sighting(
                            person_name=vehicle_desc,
                            confidence=float(conf),
                            camera_id="CAM-LIVE-01",
                            location="Grid Sector Alpha",
                            bbox=(x1, y1, x2-x1, y2-y1)
                        )
                        last_sighting_time[vehicle_desc] = current_time
                    
                    # Vehicle type
                    veh_types = {2: 'CAR', 3: 'MOTO', 5: 'BUS', 7: 'TRUCK'}
                    veh_type = veh_types.get(cls, 'VEHICLE')
                    
                    if is_target:
                        # RED for target vehicle
                        color = (0, 0, 255)
                        label = f"TARGET: {vehicle_desc}"
                        thickness = 3
                        
                        # Background for label
                        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(frame, (x1, y1-label_h-10), (x1+label_w+10, y1), color, -1)
                        cv2.putText(frame, label, (x1+5, y1-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Confidence
                        conf_text = f"{int(conf*100)}%"
                        cv2.putText(frame, conf_text, (x1, y2+20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Track history trail
                        track_history[track_id].append((int((x1+x2)/2), int((y1+y2)/2)))
                        if len(track_history[track_id]) > 30:
                            track_history[track_id].pop(0)
                        
                        # Draw trail
                        points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
                        if len(points) > 1:
                            cv2.polylines(frame, [points], False, (0, 0, 255), 3)
                    else:
                        # ORANGE for unknown vehicle
                        color = (0, 165, 255)
                        label = f"{veh_type} ID:{track_id}"
                        thickness = 2
                        cv2.putText(frame, label, (x1, y1-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Draw box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # FPS
        fps_counter += 1
        if time.time() - fps_start > 1:
            logger.info(f"Performance: {fps_counter} FPS | Tracked: {len(vehicle_tracks)} vehicles")
            fps_counter = 0
            fps_start = time.time()
        
        # Overlay
        cv2.putText(frame, "HAWKEYE VEHICLE TRACKER", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 65), 2)
        
        # Show target count
        target_count = len([v for v in vehicle_tracks.values() if v['is_target']])
        cv2.putText(frame, f"TARGETS: {target_count} | TOTAL: {len(vehicle_tracks)}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        with lock:
            output_frame = frame.copy()

def generate():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@flask_app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@flask_app.route("/")
def index():
    stats = tracker.get_statistics()
    all_vehicles = [v['description'] for v in vehicle_tracks.values()]
    target_vehicles_list = [v['description'] for v in vehicle_tracks.values() if v['is_target']]
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>HAWKEYE Vehicle Tracker</title>
        <style>
            body { background: #000; color: #0f0; font-family: 'Courier New', monospace; margin: 0; padding: 20px; }
            h1 { color: #0f0; border-bottom: 2px solid #0f0; padding-bottom: 10px; }
            h2 { color: #ff0; }
            .container { max-width: 1400px; margin: 0 auto; }
            .video-container { margin: 20px 0; }
            .video-container img { width: 100%; border: 3px solid #0f0; }
            .panel { background: #111; border: 1px solid #0f0; padding: 15px; margin: 10px 0; }
            .target { color: #f00; font-weight: bold; font-size: 1.1em; }
            button { background: #0f0; color: #000; border: none; padding: 10px 20px; 
                     font-family: 'Courier New', monospace; font-weight: bold; cursor: pointer; margin: 5px; }
            button:hover { background: #0ff; }
            button.target-btn { background: #f00; color: #fff; padding: 12px 24px; font-size: 1.1em; }
            button.target-btn:hover { background: #ff4444; }
            input { background: #222; color: #0f0; border: 2px solid #0f0; padding: 10px; 
                    font-family: 'Courier New', monospace; margin: 5px; font-size: 1em; }
            .result { background: #1a1a1a; padding: 10px; margin: 5px 0; border-left: 3px solid #0f0; }
            .legend { margin: 10px 0; padding: 15px; background: #111; border: 2px solid #ff0; }
            .legend span { display: inline-block; margin-right: 20px; font-size: 1.1em; }
            .red { color: #f00; }
            .orange { color: #ffa500; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚗 HAWKEYE VEHICLE TRACKER</h1>
            <p style="font-size: 1.1em;">Professional Vehicle Surveillance & Tracking System</p>
            
            <div class="legend">
                <span class="red">🔴 RED boxes = Target vehicles (registered)</span> | 
                <span class="orange">🟠 ORANGE boxes = Unknown vehicles</span>
            </div>
            
            <div class="video-container">
                <h2>📹 Live Vehicle Feed</h2>
                <img src="/video_feed" />
            </div>
            
            <div class="panel">
                <h2>🎯 Registered Target Vehicles</h2>
                {% if target_vehicles_list %}
                <p class="target">{{ target_vehicles_list|join(', ') }}</p>
                {% else %}
                <p style="color: #ff0;">No target vehicles registered yet</p>
                {% endif %}
                <p style="color: #0ff;">Total Vehicles Tracked: {{ all_vehicles|length }}</p>
            </div>
            
            <div class="panel">
                <h2>➕ Register Vehicle as Target</h2>
                <p style="color: #ff0; font-size: 1em;">💡 Watch the video to see vehicle Track IDs (e.g., "CAR ID:15"), then register them below</p>
                <div style="margin: 15px 0;">
                    <input type="number" id="vehicle_track_id" placeholder="Enter Track ID (e.g., 15)" style="width: 200px;" />
                    <input type="text" id="vehicle_desc" placeholder="Vehicle Description (e.g., KCA 123X White Toyota)" style="width: 400px;" />
                    <button class="target-btn" onclick="registerVehicle()">🚗 REGISTER AS TARGET</button>
                </div>
            </div>
            
            <div class="panel">
                <h2>🔍 Search & Track Vehicles</h2>
                <div style="margin: 15px 0;">
                    <input type="text" id="search_vehicle" placeholder="Enter vehicle name or ID to search..." style="width: 500px; padding: 12px; font-size: 1em;" />
                    <button onclick="searchVehicleByName()" style="padding: 12px 24px;">🔍 SEARCH</button>
                    <button onclick="showAllVehicles()" style="padding: 12px 24px;">📋 SHOW ALL</button>
                </div>
                
                <div style="margin: 15px 0;">
                    <h3 style="color: #0ff;">Quick Actions:</h3>
                    <button onclick="getStats()">📈 View Statistics</button>
                    <button onclick="showTargetsOnly()">🎯 Show Targets Only</button>
                </div>
            </div>
            
            <div class="panel" id="vehicle_list" style="display: none;">
                <h2>📋 All Tracked Vehicles</h2>
                <div id="vehicle_buttons"></div>
            </div>
            
            <div class="panel" id="results">
                <h2>📊 Results</h2>
                <p>Click a button above to view vehicle data...</p>
            </div>
        </div>
        
        <script>
            function registerVehicle() {
                const trackId = document.getElementById('vehicle_track_id').value;
                const desc = document.getElementById('vehicle_desc').value;
                
                console.log('Registering vehicle - Track ID:', trackId, 'Description:', desc);
                
                if (!trackId || !desc) {
                    alert('⚠️ Please enter both Track ID and Description');
                    return;
                }
                
                fetch('/api/register_vehicle', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({track_id: parseInt(trackId), description: desc})
                })
                .then(r => r.json())
                .then(data => {
                    alert('✓ ' + data.message);
                    location.reload();
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('❌ Error registering vehicle');
                });
            }
            
            function searchVehicle(name) {
                fetch('/api/forensic_search?person=' + encodeURIComponent(name))
                    .then(r => r.json())
                    .then(data => {
                        let html = '<h2>🔍 Vehicle Search: ' + name + '</h2>';
                        html += '<p style="color: #ff0;">Found <strong>' + data.results.length + '</strong> sightings</p>';
                        
                        const recent = data.results.slice(0, 10);
                        recent.forEach((s, i) => {
                            html += '<div class="result">';
                            html += '<div style="color: #0ff;">#' + (i+1) + ' - ' + s.timestamp + '</div>';
                            html += '<div>Camera: ' + s.camera_id + ' | Confidence: ' + (s.confidence*100).toFixed(0) + '%</div>';
                            html += '</div>';
                        });
                        
                        document.getElementById('results').innerHTML = html;
                    });
            }
            
            function getTrail(name) {
                fetch('/api/breadcrumb?person=' + encodeURIComponent(name))
                    .then(r => r.json())
                    .then(data => {
                        let html = '<h2>🗺️ Movement Trail: ' + name + '</h2>';
                        html += '<p style="color: #ff0;">' + data.trail.length + ' locations tracked</p>';
                        
                        data.trail.slice(-15).forEach((s, i) => {
                            html += '<div class="result">';
                            html += '<div style="color: #f00;">📍 ' + s.timestamp + '</div>';
                            html += '<div>' + s.camera_id + ' | ' + s.location + '</div>';
                            html += '</div>';
                        });
                        
                        document.getElementById('results').innerHTML = html;
                    });
            }
            
            function getStats() {
                fetch('/api/statistics')
                    .then(r => r.json())
                    .then(data => {
                        let html = '<h2>📈 Statistics</h2>';
                        html += '<p style="color: #ff0; font-size: 1.2em;">Total Sightings: ' + data.total_sightings + '</p>';
                        html += '<h3>Top Vehicles:</h3>';
                        
                        data.by_person.slice(0, 10).forEach((v, i) => {
                            html += '<div class="result">';
                            html += '<strong>#' + (i+1) + '</strong> ' + v.person_name + ': ' + v.count + ' sightings';
                            html += '</div>';
                        });
                        
                        document.getElementById('results').innerHTML = html;
                    });
            }
        </script>
    </body>
    </html>
    """
    
    return render_template_string(html, all_vehicles=all_vehicles, target_vehicles_list=target_vehicles_list)

@flask_app.route("/api/register_vehicle", methods=['POST'])
def api_register_vehicle():
    try:
        data = request.json
        track_id = data.get('track_id')
        description = data.get('description')
        
        logger.info(f"API: Registering vehicle - Track ID={track_id}, Description={description}")
        
        if not track_id or not description:
            return jsonify({'success': False, 'message': 'Missing track_id or description'}), 400
        
        register_vehicle_target(track_id, description)
        
        return jsonify({'success': True, 'message': f'Vehicle registered: {description}'})
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@flask_app.route("/api/forensic_search")
def api_forensic_search():
    entity = request.args.get('person')
    results = tracker.forensic_search(person_name=entity)
    return jsonify({'results': results})

@flask_app.route("/api/breadcrumb")
def api_breadcrumb():
    entity = request.args.get('person')
    trail = tracker.get_breadcrumb_trail(entity)
    return jsonify({'trail': trail})

@flask_app.route("/api/statistics")
def api_statistics():
    stats = tracker.get_statistics()
    return jsonify(stats)

if __name__ == "__main__":
    video_thread = threading.Thread(target=process_video, daemon=True)
    video_thread.start()
    
    logger.info("✓ Video processing started")
    print()
    print("=" * 70)
    print("  🚀 HAWKEYE VEHICLE TRACKER RUNNING")
    print("  📹 Dashboard: http://localhost:5001")
    print("  🚗 Vehicle-Only Tracking Mode")
    print("  🎯 Register vehicles as targets via web interface")
    print("  ⚡ Press Ctrl+C to stop")
    print("=" * 70)
    print()
    
    flask_app.run(host="0.0.0.0", port=5001, debug=False, threaded=True, use_reloader=False)
