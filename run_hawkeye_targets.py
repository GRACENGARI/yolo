"""
HAWKEYE CV-Engine - Multi-Target Tracking System
Track specific PEOPLE and VEHICLES as known targets with RED boxes
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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("HAWKEYE_TARGETS")

# Flask app
flask_app = Flask(__name__)
output_frame = None
lock = threading.Lock()

# Configuration
DEVICE = os.getenv('DEVICE', 'cpu')
VIDEO_SOURCE = os.getenv('VIDEO_SOURCE', 'people.mp4')
DETECTION_THRESHOLD = float(os.getenv('DETECTION_THRESHOLD', '0.4'))

# TARGET CONFIGURATION - Edit these to set your targets
PERSON_TARGETS = {
    # Track ID will be assigned automatically to first person
    # Format: track_id: "Name"
    # Example: 1: "Juma Macharia", 5: "Suspect Alpha"
}

VEHICLE_TARGETS = {
    # Track ID: "Vehicle Description"
    # Example: 10: "KCA 123X White Toyota", 15: "KBZ 456Y Black Nissan"
}

print("=" * 70)
print("  HAWKEYE CV-Engine // MULTI-TARGET TRACKING")
print("  Track Specific People & Vehicles")
print("=" * 70)
print(f"  Device: {DEVICE}")
print(f"  Video Source: {VIDEO_SOURCE}")
print("=" * 70)
print()

# Initialize
logger.info("Initializing YOLO...")
model = YOLO("yolov8n.pt")
model.to(DEVICE)

logger.info("Initializing Sighting Tracker...")
tracker = SightingTracker()

# Tracking state
person_tracks = {}  # {track_id: {'name': str, 'is_target': bool, ...}}
vehicle_tracks = {}  # {track_id: {'description': str, 'is_target': bool, ...}}
next_person_id = 1
next_vehicle_id = 1
first_person_assigned = False
last_sighting_time = {}

def get_person_info(track_id):
    """Get or create person info for track ID"""
    global next_person_id, first_person_assigned
    
    if track_id not in person_tracks:
        # Check if this track_id is a registered target
        if track_id in PERSON_TARGETS:
            name = PERSON_TARGETS[track_id]
            is_target = True
            logger.info(f"🎯 REGISTERED TARGET DETECTED: {name} (Track ID: {track_id})")
        elif not first_person_assigned:
            # First person is automatically "Juma Macharia"
            name = "Juma Macharia"
            is_target = True
            first_person_assigned = True
            PERSON_TARGETS[track_id] = name
            logger.info(f"🎯 AUTO-TARGET: {name} (Track ID: {track_id})")
        else:
            # Other people get generic names
            name = f"Person-{next_person_id}"
            is_target = False
            next_person_id += 1
        
        person_tracks[track_id] = {
            'name': name,
            'is_target': is_target,
            'first_seen': time.time(),
            'last_seen': time.time(),
            'sightings': 0
        }
    
    person_tracks[track_id]['last_seen'] = time.time()
    person_tracks[track_id]['sightings'] += 1
    
    return person_tracks[track_id]

def get_vehicle_info(track_id):
    """Get or create vehicle info for track ID"""
    global next_vehicle_id
    
    if track_id not in vehicle_tracks:
        # Check if this track_id is a registered target vehicle
        if track_id in VEHICLE_TARGETS:
            description = VEHICLE_TARGETS[track_id]
            is_target = True
            logger.info(f"🚗 TARGET VEHICLE DETECTED: {description} (Track ID: {track_id})")
        else:
            # Other vehicles get generic IDs
            description = f"Vehicle-{next_vehicle_id}"
            is_target = False
            next_vehicle_id += 1
        
        vehicle_tracks[track_id] = {
            'description': description,
            'is_target': is_target,
            'first_seen': time.time(),
            'last_seen': time.time(),
            'sightings': 0
        }
    
    vehicle_tracks[track_id]['last_seen'] = time.time()
    vehicle_tracks[track_id]['sightings'] += 1
    
    return vehicle_tracks[track_id]

def register_person_target(track_id, name):
    """Register a person as a target"""
    PERSON_TARGETS[track_id] = name
    if track_id in person_tracks:
        person_tracks[track_id]['name'] = name
        person_tracks[track_id]['is_target'] = True
    logger.info(f"✓ Registered person target: {name} (Track ID: {track_id})")

def register_vehicle_target(track_id, description):
    """Register a vehicle as a target"""
    VEHICLE_TARGETS[track_id] = description
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
        results = model.track(frame, persist=True, verbose=False, device=DEVICE)
        
        # Process detections
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, track_id, conf, cls in zip(boxes, track_ids, confidences, classes):
                x1, y1, x2, y2 = map(int, box)
                
                # PERSON DETECTION
                if cls == 0 and conf > DETECTION_THRESHOLD:
                    person_info = get_person_info(track_id)
                    person_name = person_info['name']
                    is_target = person_info['is_target']
                    
                    # Record sighting (throttled)
                    current_time = time.time()
                    last_time = last_sighting_time.get(f"person_{person_name}", 0)
                    if current_time - last_time > 2:
                        tracker.add_sighting(
                            person_name=person_name,
                            confidence=float(conf),
                            camera_id="CAM-LIVE-01",
                            location="Grid Sector Alpha",
                            bbox=(x1, y1, x2-x1, y2-y1)
                        )
                        last_sighting_time[f"person_{person_name}"] = current_time
                    
                    # Visual styling
                    if is_target:
                        # RED for target person
                        color = (0, 0, 255)
                        label = f"🎯 TARGET: {person_name}"
                        thickness = 3
                        
                        # Background for label
                        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(frame, (x1, y1-label_h-15), (x1+label_w+10, y1), color, -1)
                        cv2.putText(frame, label, (x1+5, y1-8),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Confidence bar
                        bar_width = int((x2 - x1) * conf)
                        cv2.rectangle(frame, (x1, y2+5), (x1+bar_width, y2+15), (0, 0, 255), -1)
                        
                        # Track history trail
                        track_history[f"person_{track_id}"].append((int((x1+x2)/2), int((y1+y2)/2)))
                        if len(track_history[f"person_{track_id}"]) > 30:
                            track_history[f"person_{track_id}"].pop(0)
                        
                        # Draw trail
                        points = np.array(track_history[f"person_{track_id}"], dtype=np.int32).reshape((-1, 1, 2))
                        if len(points) > 1:
                            cv2.polylines(frame, [points], False, (0, 0, 255), 2)
                    else:
                        # GREEN for unknown person
                        color = (0, 255, 65)
                        label = f"{person_name} (ID:{track_id})"
                        thickness = 2
                        cv2.putText(frame, label, (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Draw box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # VEHICLE DETECTION
                elif cls in [2, 3, 5, 7] and conf > DETECTION_THRESHOLD:
                    vehicle_info = get_vehicle_info(track_id)
                    vehicle_desc = vehicle_info['description']
                    is_target = vehicle_info['is_target']
                    
                    # Record vehicle sighting
                    current_time = time.time()
                    last_time = last_sighting_time.get(f"vehicle_{vehicle_desc}", 0)
                    if current_time - last_time > 2:
                        tracker.add_sighting(
                            person_name=vehicle_desc,
                            confidence=float(conf),
                            camera_id="CAM-LIVE-01",
                            location="Grid Sector Alpha",
                            bbox=(x1, y1, x2-x1, y2-y1)
                        )
                        last_sighting_time[f"vehicle_{vehicle_desc}"] = current_time
                    
                    # Visual styling
                    veh_types = {2: 'CAR', 3: 'MOTO', 5: 'BUS', 7: 'TRUCK'}
                    veh_type = veh_types.get(cls, 'VEHICLE')
                    
                    if is_target:
                        # RED for target vehicle
                        color = (0, 0, 255)
                        label = f"🚗 TARGET: {vehicle_desc}"
                        thickness = 3
                        
                        # Background for label
                        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(frame, (x1, y1-label_h-15), (x1+label_w+10, y1), color, -1)
                        cv2.putText(frame, label, (x1+5, y1-8),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Track history trail
                        track_history[f"vehicle_{track_id}"].append((int((x1+x2)/2), int((y1+y2)/2)))
                        if len(track_history[f"vehicle_{track_id}"]) > 30:
                            track_history[f"vehicle_{track_id}"].pop(0)
                        
                        # Draw trail
                        points = np.array(track_history[f"vehicle_{track_id}"], dtype=np.int32).reshape((-1, 1, 2))
                        if len(points) > 1:
                            cv2.polylines(frame, [points], False, (0, 0, 255), 2)
                    else:
                        # ORANGE for unknown vehicle
                        color = (0, 165, 255)
                        label = f"{veh_type}: {vehicle_desc} (ID:{track_id})"
                        thickness = 2
                        cv2.putText(frame, label, (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Draw box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # FPS
        fps_counter += 1
        if time.time() - fps_start > 1:
            logger.info(f"Performance: {fps_counter} FPS | People: {len(person_tracks)} | Vehicles: {len(vehicle_tracks)}")
            fps_counter = 0
            fps_start = time.time()
        
        # Overlay
        cv2.putText(frame, "HAWKEYE // MULTI-TARGET TRACKING", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 65), 2)
        
        # Show registered targets
        target_people = [p['name'] for p in person_tracks.values() if p['is_target']]
        target_vehicles = [v['description'] for v in vehicle_tracks.values() if v['is_target']]
        
        y_pos = 60
        if target_people:
            cv2.putText(frame, f"PERSON TARGETS: {', '.join(target_people)}", (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            y_pos += 25
        
        if target_vehicles:
            cv2.putText(frame, f"VEHICLE TARGETS: {', '.join(target_vehicles)}", (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
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
    target_people = [p['name'] for p in person_tracks.values() if p['is_target']]
    target_vehicles = [v['description'] for v in vehicle_tracks.values() if v['is_target']]
    all_people = [p['name'] for p in person_tracks.values()]
    all_vehicles = [v['description'] for v in vehicle_tracks.values()]
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>HAWKEYE Multi-Target System</title>
        <style>
            body { background: #000; color: #0f0; font-family: 'Courier New', monospace; margin: 0; padding: 20px; }
            h1 { color: #0f0; border-bottom: 2px solid #0f0; padding-bottom: 10px; }
            h2 { color: #ff0; }
            .container { max-width: 1400px; margin: 0 auto; }
            .video-container { margin: 20px 0; }
            .video-container img { width: 100%; border: 2px solid #0f0; }
            .panel { background: #111; border: 1px solid #0f0; padding: 15px; margin: 10px 0; }
            .target { color: #f00; font-weight: bold; font-size: 1.1em; }
            .stat { color: #ff0; }
            button { background: #0f0; color: #000; border: none; padding: 10px 20px; 
                     font-family: 'Courier New', monospace; font-weight: bold; cursor: pointer; margin: 5px; }
            button:hover { background: #0ff; }
            button.target-btn { background: #f00; color: #fff; }
            button.target-btn:hover { background: #ff4444; }
            .result { background: #222; padding: 10px; margin: 5px 0; border-left: 3px solid #0f0; }
            .breadcrumb { background: #1a1a1a; padding: 8px; margin: 3px 0; border-left: 3px solid #f00; }
            .legend { margin: 10px 0; padding: 10px; background: #111; }
            .legend span { display: inline-block; margin-right: 20px; }
            .red { color: #f00; }
            .green { color: #0f0; }
            .orange { color: #ffa500; }
            input { background: #222; color: #0f0; border: 1px solid #0f0; padding: 8px; 
                    font-family: 'Courier New', monospace; margin: 5px; }
            .section { margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 HAWKEYE MULTI-TARGET TRACKING SYSTEM</h1>
            <p>Track Specific People & Vehicles with RED Boxes</p>
            
            <div class="legend">
                <span class="red">🔴 RED boxes = Known targets (People & Vehicles)</span> | 
                <span class="green">🟢 GREEN boxes = Unknown persons</span> | 
                <span class="orange">🟠 ORANGE boxes = Unknown vehicles</span>
            </div>
            
            <div class="video-container">
                <h2>📹 Live Video Feed</h2>
                <img src="/video_feed" />
            </div>
            
            <div class="panel">
                <h2>🎯 Registered Targets</h2>
                <div class="section">
                    <h3 class="target">PERSON TARGETS:</h3>
                    <p class="target">{{ target_people|join(', ') or 'None registered yet' }}</p>
                </div>
                <div class="section">
                    <h3 class="target">VEHICLE TARGETS:</h3>
                    <p class="target">{{ target_vehicles|join(', ') or 'None registered yet' }}</p>
                </div>
                <p class="stat">Total Sightings: {{ stats['total_sightings'] }}</p>
            </div>
            
            <div class="panel">
                <h2>➕ Register New Targets</h2>
                <div class="section">
                    <h3>Register Person as Target:</h3>
                    <input type="number" id="person_track_id" placeholder="Track ID" />
                    <input type="text" id="person_name" placeholder="Person Name" />
                    <button class="target-btn" onclick="registerPerson()">🎯 Register Person</button>
                </div>
                <div class="section">
                    <h3>Register Vehicle as Target:</h3>
                    <input type="number" id="vehicle_track_id" placeholder="Track ID" />
                    <input type="text" id="vehicle_desc" placeholder="Vehicle Description (e.g., KCA 123X White Toyota)" />
                    <button class="target-btn" onclick="registerVehicle()">🚗 Register Vehicle</button>
                </div>
                <p style="color: #ff0; font-size: 0.9em;">💡 Tip: Watch the video feed to see Track IDs, then register them as targets</p>
            </div>
            
            <div class="panel">
                <h2>🔍 Forensic Search & Breadcrumb Trails</h2>
                <h3>People:</h3>
                {% for person in all_people %}
                <button onclick="searchEntity('{{ person }}')">🔍 {{ person }}</button>
                <button onclick="getBreadcrumb('{{ person }}')">🗺️ Trail</button>
                {% endfor %}
                
                <h3>Vehicles:</h3>
                {% for vehicle in all_vehicles %}
                <button onclick="searchEntity('{{ vehicle }}')">🔍 {{ vehicle }}</button>
                <button onclick="getBreadcrumb('{{ vehicle }}')">🗺️ Trail</button>
                {% endfor %}
                
                <br><br>
                <button onclick="getRecent()">📊 Recent Sightings</button>
                <button onclick="getStats()">📈 Statistics</button>
            </div>
            
            <div class="panel" id="results">
                <h2>📊 Results</h2>
                <p>Click a button above to view forensic data...</p>
            </div>
        </div>
        
        <script>
            function registerPerson() {
                const trackId = document.getElementById('person_track_id').value;
                const name = document.getElementById('person_name').value;
                
                if (!trackId || !name) {
                    alert('Please enter both Track ID and Name');
                    return;
                }
                
                fetch('/api/register_person', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({track_id: parseInt(trackId), name: name})
                })
                .then(r => r.json())
                .then(data => {
                    alert(data.message);
                    location.reload();
                });
            }
            
            function registerVehicle() {
                const trackId = document.getElementById('vehicle_track_id').value;
                const desc = document.getElementById('vehicle_desc').value;
                
                if (!trackId || !desc) {
                    alert('Please enter both Track ID and Description');
                    return;
                }
                
                fetch('/api/register_vehicle', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({track_id: parseInt(trackId), description: desc})
                })
                .then(r => r.json())
                .then(data => {
                    alert(data.message);
                    location.reload();
                });
            }
            
            function searchEntity(name) {
                fetch('/api/forensic_search?entity=' + encodeURIComponent(name))
                    .then(r => r.json())
                    .then(data => {
                        let html = '<h2>🔍 Forensic Search: ' + name + '</h2>';
                        html += '<p>Found <strong>' + data.results.length + '</strong> sightings</p>';
                        data.results.forEach(s => {
                            html += '<div class="result">';
                            html += '<strong>' + s.timestamp + '</strong><br>';
                            html += 'Camera: ' + s.camera_id + ' | Location: ' + s.location + '<br>';
                            html += 'Confidence: ' + (s.confidence * 100).toFixed(0) + '%';
                            html += '</div>';
                        });
                        document.getElementById('results').innerHTML = html;
                    });
            }
            
            function getBreadcrumb(name) {
                fetch('/api/breadcrumb?entity=' + encodeURIComponent(name))
                    .then(r => r.json())
                    .then(data => {
                        let html = '<h2>🗺️ Breadcrumb Trail: ' + name + '</h2>';
                        html += '<p><strong>' + data.trail.length + '</strong> locations tracked</p>';
                        data.trail.forEach((s, i) => {
                            html += '<div class="breadcrumb">';
                            html += '<strong>Step ' + (i+1) + ':</strong> ' + s.timestamp + '<br>';
                            html += 'Camera: ' + s.camera_id + ' | Location: ' + s.location;
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
                            html += '<div class="result">';
                            html += '<strong>' + s.person_name + '</strong> - ' + s.timestamp + '<br>';
                            html += 'Camera: ' + s.camera_id;
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
                        html += '<p>Total Sightings: <strong>' + data.total_sightings + '</strong></p>';
                        html += '<h3>By Entity:</h3>';
                        data.by_person.forEach(p => {
                            html += '<div class="result">';
                            html += p.person_name + ': <strong>' + p.count + '</strong> sightings';
                            html += '</div>';
                        });
                        document.getElementById('results').innerHTML = html;
                    });
            }
        </script>
    </body>
    </html>
    """
    
    return render_template_string(html, 
                                 target_people=target_people,
                                 target_vehicles=target_vehicles,
                                 all_people=all_people,
                                 all_vehicles=all_vehicles,
                                 stats=stats)

@flask_app.route("/api/register_person", methods=['POST'])
def api_register_person():
    data = request.json
    track_id = data.get('track_id')
    name = data.get('name')
    register_person_target(track_id, name)
    return jsonify({'success': True, 'message': f'Registered {name} as target'})

@flask_app.route("/api/register_vehicle", methods=['POST'])
def api_register_vehicle():
    data = request.json
    track_id = data.get('track_id')
    description = data.get('description')
    register_vehicle_target(track_id, description)
    return jsonify({'success': True, 'message': f'Registered {description} as target'})

@flask_app.route("/api/forensic_search")
def api_forensic_search():
    entity = request.args.get('entity')
    results = tracker.forensic_search(person_name=entity)
    return jsonify({'results': results})

@flask_app.route("/api/breadcrumb")
def api_breadcrumb():
    entity = request.args.get('entity')
    trail = tracker.get_breadcrumb_trail(entity)
    return jsonify({'trail': trail})

@flask_app.route("/api/recent_sightings")
def api_recent_sightings():
    sightings = tracker.get_recent_sightings(limit=50)
    return jsonify({'sightings': sightings})

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
    print("  🚀 HAWKEYE MULTI-TARGET SYSTEM RUNNING")
    print("  📹 Dashboard: http://localhost:5001")
    print("  🎯 First person = 'Juma Macharia' (AUTO-TARGET)")
    print("  ➕ Register more targets via web interface")
    print("  🔍 Forensic Search: ENABLED")
    print("  🗺️ Breadcrumb Trails: ENABLED")
    print("  ⚡ Press Ctrl+C to stop")
    print("=" * 70)
    print()
    
    flask_app.run(host="0.0.0.0", port=5001, debug=False, threaded=True, use_reloader=False)
