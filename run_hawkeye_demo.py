"""
HAWKEYE CV-Engine - Working Demo
Person tracking with ID-based identification, forensic search, and breadcrumb trails
Works without face detection - uses tracking IDs
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

# Load environment variables
from pathlib import Path
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
logger = logging.getLogger("HAWKEYE_DEMO")

# Flask app
flask_app = Flask(__name__)
output_frame = None
lock = threading.Lock()

# Configuration
DEVICE = os.getenv('DEVICE', 'cpu')
VIDEO_SOURCE = os.getenv('VIDEO_SOURCE', 'people.mp4')
DETECTION_THRESHOLD = float(os.getenv('DETECTION_THRESHOLD', '0.6'))  # Increased from 0.4 to 0.6

print("=" * 70)
print("  HAWKEYE CV-Engine // WORKING DEMO")
print("  Person Tracking + Forensic Search + Breadcrumb Trails")
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
person_tracks = {}  # {track_id: {'name': str, 'first_seen': time, 'last_seen': time, 'sightings': int}}
vehicle_tracks = {}  # {track_id: {'description': str, 'is_target': bool, ...}}
next_person_id = 1
next_vehicle_id = 1
target_person = None  # Will be set to first person detected
target_vehicles = {}  # {track_id: description} - Vehicles marked as targets
last_sighting_time = {}

def get_person_name(track_id):
    """Get or create person name for track ID"""
    global next_person_id, target_person
    
    if track_id not in person_tracks:
        if target_person is None:
            # First person is "Juma Macharia"
            name = "Juma Macharia"
            target_person = track_id
            logger.info(f"🎯 TARGET IDENTIFIED: {name} (Track ID: {track_id})")
        else:
            # Other people get generic names
            name = f"Person-{next_person_id}"
            next_person_id += 1
        
        person_tracks[track_id] = {
            'name': name,
            'first_seen': time.time(),
            'last_seen': time.time(),
            'sightings': 0
        }
    
    person_tracks[track_id]['last_seen'] = time.time()
    person_tracks[track_id]['sightings'] += 1
    
    return person_tracks[track_id]['name']

def get_vehicle_info(track_id):
    """Get or create vehicle info for track ID"""
    global next_vehicle_id
    
    if track_id not in vehicle_tracks:
        description = f"Vehicle-{next_vehicle_id}"
        is_target = track_id in target_vehicles
        next_vehicle_id += 1
        
        vehicle_tracks[track_id] = {
            'description': target_vehicles.get(track_id, description),
            'is_target': is_target,
            'first_seen': time.time(),
            'last_seen': time.time(),
            'sightings': 0
        }
        
        if is_target:
            logger.info(f"🚗 TARGET VEHICLE: {vehicle_tracks[track_id]['description']} (Track ID: {track_id})")
    
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
    frame_count = 0
    
    # Simple tracking
    from collections import defaultdict
    track_history = defaultdict(lambda: [])
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        frame_count += 1
        
        # Run YOLO with tracking
        results = model.track(frame, persist=True, verbose=False, device=DEVICE)
        
        # Process detections
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, track_id, conf, cls in zip(boxes, track_ids, confidences, classes):
                if cls == 0 and conf > DETECTION_THRESHOLD:  # Person
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Get person name
                    person_name = get_person_name(track_id)
                    
                    # Record sighting (throttled)
                    current_time = time.time()
                    last_time = last_sighting_time.get(person_name, 0)
                    if current_time - last_time > 2:  # Every 2 seconds
                        tracker.add_sighting(
                            person_name=person_name,
                            confidence=float(conf),
                            camera_id="CAM-LIVE-01",
                            location="Grid Sector Alpha",
                            bbox=(x1, y1, x2-x1, y2-y1)
                        )
                        last_sighting_time[person_name] = current_time
                    
                    # Visual styling
                    is_target = (track_id == target_person)
                    
                    if is_target:
                        # RED for target
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
                        track_history[track_id].append((int((x1+x2)/2), int((y1+y2)/2)))
                        if len(track_history[track_id]) > 30:
                            track_history[track_id].pop(0)
                        
                        # Draw trail
                        points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
                        if len(points) > 1:
                            cv2.polylines(frame, [points], False, (0, 0, 255), 2)
                    else:
                        # GREEN for others - SIMPLIFIED
                        color = (0, 255, 65)
                        label = f"ID:{track_id}"  # Just show ID
                        thickness = 2
                        cv2.putText(frame, label, (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)  # Smaller font
                    
                    # Draw box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                elif cls in [2, 3, 5, 7] and conf > DETECTION_THRESHOLD:  # Vehicles
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Get vehicle info
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
                        # ORANGE for unknown vehicle - SIMPLIFIED LABEL
                        color = (0, 165, 255)
                        label = f"ID:{track_id}"  # Just show ID, not full description
                        thickness = 2
                        cv2.putText(frame, label, (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)  # Smaller font
                    
                    # Draw box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # FPS
        fps_counter += 1
        if time.time() - fps_start > 1:
            logger.info(f"Performance: {fps_counter} FPS | Tracked: {len(person_tracks)} people")
            fps_counter = 0
            fps_start = time.time()
        
        # Overlay
        cv2.putText(frame, "HAWKEYE DEMO // TARGET TRACKING ACTIVE", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 65), 2)
        
        if target_person is not None:
            target_name = person_tracks[target_person]['name']
            cv2.putText(frame, f"TARGET: {target_name}", (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        stats = tracker.get_statistics()
        cv2.putText(frame, f"SIGHTINGS: {stats['total_sightings']}", (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
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
    tracked_people = [p['name'] for p in person_tracks.values()]
    tracked_vehicles = [v['description'] for v in vehicle_tracks.values()]
    target_vehicles_list = [v['description'] for v in vehicle_tracks.values() if v['is_target']]
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>HAWKEYE Demo System</title>
        <style>
            body { background: #000; color: #0f0; font-family: 'Courier New', monospace; margin: 0; padding: 20px; }
            h1 { color: #0f0; border-bottom: 2px solid #0f0; padding-bottom: 10px; }
            h2 { color: #ff0; }
            .container { max-width: 1400px; margin: 0 auto; }
            .video-container { margin: 20px 0; }
            .video-container img { width: 100%; border: 2px solid #0f0; }
            .panel { background: #111; border: 1px solid #0f0; padding: 15px; margin: 10px 0; }
            .target { color: #f00; font-weight: bold; font-size: 1.2em; }
            .stat { color: #ff0; }
            button { background: #0f0; color: #000; border: none; padding: 10px 20px; 
                     font-family: 'Courier New', monospace; font-weight: bold; cursor: pointer; margin: 5px; }
            button:hover { background: #0ff; }
            .result { background: #222; padding: 10px; margin: 5px 0; border-left: 3px solid #0f0; }
            .breadcrumb { background: #1a1a1a; padding: 8px; margin: 3px 0; border-left: 3px solid #f00; }
            .legend { margin: 10px 0; padding: 10px; background: #111; }
            .legend span { display: inline-block; margin-right: 20px; }
            .red { color: #f00; }
            .green { color: #0f0; }
            .orange { color: #ffa500; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 HAWKEYE DEMO SYSTEM</h1>
            <p>Person Tracking + Forensic Search + Breadcrumb Trails</p>
            
            <div class="legend">
                <span class="red">🔴 RED boxes = Known targets</span> | 
                <span class="green">🟢 GREEN boxes = Unknown persons</span> | 
                <span class="orange">🟠 ORANGE boxes = Unknown vehicles</span>
            </div>
            
            <div class="video-container">
                <h2>📹 Live Video Feed</h2>
                <img src="/video_feed" />
            </div>
            
            <div class="panel">
                <h2>🎯 Tracked People</h2>
                <p class="target">{{ tracked_people|join(', ') or 'Waiting for detections...' }}</p>
                <p class="stat">Total Sightings Recorded: {{ stats['total_sightings'] }}</p>
            </div>
            
            <div class="panel">
                <h2>🚗 Tracked Vehicles</h2>
                <p>{{ tracked_vehicles|join(', ') or 'No vehicles detected yet' }}</p>
                {% if target_vehicles_list %}
                <p class="target">TARGETS: {{ target_vehicles_list|join(', ') }}</p>
                {% endif %}
            </div>
            
            <div class="panel">
                <h2>➕ Register Vehicle as Target</h2>
                <p style="color: #ff0; font-size: 0.9em;">💡 Watch the video to see vehicle Track IDs, then register them as targets</p>
                <input type="number" id="vehicle_track_id" placeholder="Track ID" style="background: #222; color: #0f0; border: 1px solid #0f0; padding: 8px; font-family: 'Courier New', monospace; margin: 5px;" />
                <input type="text" id="vehicle_desc" placeholder="Vehicle Description (e.g., KCA 123X White Toyota)" style="background: #222; color: #0f0; border: 1px solid #0f0; padding: 8px; font-family: 'Courier New', monospace; margin: 5px; width: 300px;" />
                <button onclick="registerVehicle(); return false;" style="background: #f00; color: #fff; padding: 10px 20px; border: none; cursor: pointer; font-family: 'Courier New', monospace; font-weight: bold;">🚗 Register Vehicle Target</button>
            </div>
            
            <div class="panel">
                <h2>🔍 Forensic Search & Breadcrumb Trails</h2>
                <h3>People:</h3>
                {% for person in tracked_people %}
                <button onclick="searchPerson('{{ person }}')">🔍 Search: {{ person }}</button>
                <button onclick="getBreadcrumb('{{ person }}')">🗺️ Trail: {{ person }}</button>
                {% endfor %}
                
                <h3>Vehicles:</h3>
                {% for vehicle in tracked_vehicles %}
                <button onclick="searchPerson('{{ vehicle }}')">🔍 Search: {{ vehicle }}</button>
                <button onclick="getBreadcrumb('{{ vehicle }}')">🗺️ Trail: {{ vehicle }}</button>
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
            function registerVehicle() {
                console.log('registerVehicle function called');
                const trackId = document.getElementById('vehicle_track_id').value;
                const desc = document.getElementById('vehicle_desc').value;
                
                console.log('Track ID:', trackId, 'Description:', desc);
                
                if (!trackId || !desc) {
                    alert('Please enter both Track ID and Description');
                    return;
                }
                
                console.log('Sending request to /api/register_vehicle');
                
                fetch('/api/register_vehicle', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({track_id: parseInt(trackId), description: desc})
                })
                .then(r => {
                    console.log('Response received:', r);
                    return r.json();
                })
                .then(data => {
                    console.log('Response data:', data);
                    alert(data.message);
                    location.reload();
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('Error registering vehicle: ' + error);
                });
            }
            
            function searchPerson(name) {
                fetch('/api/forensic_search?person=' + encodeURIComponent(name))
                    .then(r => r.json())
                    .then(data => {
                        let html = '<h2>🔍 Forensic Search: ' + name + '</h2>';
                        html += '<p style="color: #ff0;">Found <strong>' + data.results.length + '</strong> total sightings</p>';
                        
                        if (data.results.length === 0) {
                            html += '<p>No sightings recorded yet.</p>';
                        } else {
                            // Show only last 10 sightings
                            const recentResults = data.results.slice(0, 10);
                            html += '<p style="color: #0ff;">Showing most recent 10 sightings:</p>';
                            
                            recentResults.forEach((s, index) => {
                                html += '<div class="result" style="margin: 10px 0; padding: 10px; background: #1a1a1a; border-left: 3px solid #0f0;">';
                                html += '<div style="color: #0ff; font-weight: bold;">Sighting #' + (index + 1) + '</div>';
                                html += '<div style="margin: 5px 0;"><span style="color: #ff0;">Time:</span> ' + s.timestamp + '</div>';
                                html += '<div style="margin: 5px 0;"><span style="color: #ff0;">Camera:</span> ' + s.camera_id + '</div>';
                                html += '<div style="margin: 5px 0;"><span style="color: #ff0;">Location:</span> ' + s.location + '</div>';
                                html += '<div style="margin: 5px 0;"><span style="color: #ff0;">Confidence:</span> ' + (s.confidence * 100).toFixed(0) + '%</div>';
                                html += '</div>';
                            });
                            
                            if (data.results.length > 10) {
                                html += '<p style="color: #f90; margin-top: 15px;">⚠️ ' + (data.results.length - 10) + ' more sightings not shown. Use breadcrumb trail for complete history.</p>';
                            }
                        }
                        
                        document.getElementById('results').innerHTML = html;
                    });
            }
                            html += '</div>';
                        });
                        document.getElementById('results').innerHTML = html;
                    });
            }
            
            function getBreadcrumb(name) {
                fetch('/api/breadcrumb?person=' + encodeURIComponent(name))
                    .then(r => r.json())
                    .then(data => {
                        let html = '<h2>🗺️ Breadcrumb Trail: ' + name + '</h2>';
                        
                        if (data.trail.length === 0) {
                            html += '<p>No movement data recorded yet.</p>';
                        } else {
                            html += '<p style="color: #ff0;"><strong>' + data.trail.length + '</strong> locations tracked</p>';
                            
                            // Show only last 15 locations
                            const recentTrail = data.trail.slice(-15);
                            html += '<p style="color: #0ff;">Showing most recent 15 locations:</p>';
                            
                            recentTrail.forEach((s, i) => {
                                const stepNum = data.trail.length - recentTrail.length + i + 1;
                                html += '<div class="breadcrumb" style="margin: 8px 0; padding: 10px; background: #1a1a1a; border-left: 3px solid #f00;">';
                                html += '<div style="color: #f00; font-weight: bold;">📍 Step ' + stepNum + '</div>';
                                html += '<div style="margin: 5px 0;"><span style="color: #ff0;">Time:</span> ' + s.timestamp + '</div>';
                                html += '<div style="margin: 5px 0;"><span style="color: #ff0;">Camera:</span> ' + s.camera_id + '</div>';
                                html += '<div style="margin: 5px 0;"><span style="color: #ff0;">Location:</span> ' + s.location + '</div>';
                                html += '</div>';
                            });
                            
                            if (data.trail.length > 15) {
                                html += '<p style="color: #f90; margin-top: 15px;">⚠️ Showing last 15 of ' + data.trail.length + ' total locations</p>';
                            }
                            
                            // Summary
                            html += '<div style="margin-top: 20px; padding: 15px; background: #0a0a0a; border: 1px solid #0f0;">';
                            html += '<h3 style="color: #0f0;">📊 Movement Summary</h3>';
                            html += '<p><span style="color: #ff0;">First Seen:</span> ' + data.trail[0].timestamp + '</p>';
                            html += '<p><span style="color: #ff0;">Last Seen:</span> ' + data.trail[data.trail.length - 1].timestamp + '</p>';
                            html += '<p><span style="color: #ff0;">Total Locations:</span> ' + data.trail.length + '</p>';
                            html += '</div>';
                        }
                        
                        document.getElementById('results').innerHTML = html;
                    });
            }
            
            function getRecent() {
                fetch('/api/recent_sightings')
                    .then(r => r.json())
                    .then(data => {
                        let html = '<h2>📊 Recent Sightings</h2>';
                        html += '<p style="color: #ff0;">Last 20 detections across all targets</p>';
                        
                        const recentSightings = data.sightings.slice(0, 20);
                        recentSightings.forEach((s, index) => {
                            html += '<div class="result" style="margin: 8px 0; padding: 8px; background: #1a1a1a; border-left: 3px solid #0f0;">';
                            html += '<div style="color: #0ff;">#' + (index + 1) + ' - <strong>' + s.person_name + '</strong></div>';
                            html += '<div style="font-size: 0.9em; color: #aaa;">' + s.timestamp + ' | ' + s.camera_id + '</div>';
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
                        html += '<div style="padding: 15px; background: #0a0a0a; border: 1px solid #0f0; margin: 10px 0;">';
                        html += '<p style="color: #ff0; font-size: 1.2em;">Total Sightings: <strong style="color: #0f0;">' + data.total_sightings + '</strong></p>';
                        html += '</div>';
                        
                        html += '<h3 style="color: #0ff; margin-top: 20px;">Top 10 Most Detected:</h3>';
                        const top10 = data.by_person.slice(0, 10);
                        top10.forEach((p, index) => {
                            const barWidth = (p.count / data.by_person[0].count) * 100;
                            html += '<div class="result" style="margin: 8px 0; padding: 10px; background: #1a1a1a;">';
                            html += '<div style="color: #0f0; font-weight: bold;">#' + (index + 1) + ' - ' + p.person_name + '</div>';
                            html += '<div style="margin: 5px 0;">';
                            html += '<div style="background: #0f0; height: 20px; width: ' + barWidth + '%; display: inline-block;"></div>';
                            html += '<span style="margin-left: 10px; color: #ff0;">' + p.count + ' sightings</span>';
                            html += '</div>';
                            html += '</div>';
                        });
                        
                        if (data.by_person.length > 10) {
                            html += '<p style="color: #f90; margin-top: 10px;">+ ' + (data.by_person.length - 10) + ' more entities tracked</p>';
                        }
                        
                        document.getElementById('results').innerHTML = html;
                    });
            }
            
            // Auto-refresh stats every 5 seconds
            setInterval(() => {
                fetch('/api/statistics')
                    .then(r => r.json())
                    .then(data => {
                        // Update sighting count if visible
                    });
            }, 5000);
        </script>
    </body>
    </html>
    """
    
    return render_template_string(html, tracked_people=tracked_people, tracked_vehicles=tracked_vehicles, 
                                 target_vehicles_list=target_vehicles_list, stats=stats)

@flask_app.route("/api/forensic_search")
def api_forensic_search():
    person = request.args.get('person')
    results = tracker.forensic_search(person_name=person)
    return jsonify({'results': results})

@flask_app.route("/api/breadcrumb")
def api_breadcrumb():
    person = request.args.get('person')
    trail = tracker.get_breadcrumb_trail(person)
    return jsonify({'trail': trail})

@flask_app.route("/api/recent_sightings")
def api_recent_sightings():
    sightings = tracker.get_recent_sightings(limit=50)
    return jsonify({'sightings': sightings})

@flask_app.route("/api/statistics")
def api_statistics():
    stats = tracker.get_statistics()
    return jsonify(stats)

@flask_app.route("/api/register_vehicle", methods=['POST'])
def api_register_vehicle():
    try:
        data = request.json
        track_id = data.get('track_id')
        description = data.get('description')
        
        logger.info(f"Attempting to register vehicle: Track ID={track_id}, Description={description}")
        
        if not track_id or not description:
            return jsonify({'success': False, 'message': 'Missing track_id or description'}), 400
        
        register_vehicle_target(track_id, description)
        logger.info(f"✓ Successfully registered vehicle target: {description} (Track ID: {track_id})")
        
        return jsonify({'success': True, 'message': f'✓ Registered {description} as target vehicle'})
    except Exception as e:
        logger.error(f"Error registering vehicle: {e}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

if __name__ == "__main__":
    video_thread = threading.Thread(target=process_video, daemon=True)
    video_thread.start()
    
    logger.info("✓ Video processing started")
    print()
    print("=" * 70)
    print("  🚀 HAWKEYE DEMO SYSTEM RUNNING")
    print("  📹 Dashboard: http://localhost:5001")
    print("  🎯 First person detected = 'Juma Macharia' (RED box)")
    print("  🔍 Forensic Search: ENABLED")
    print("  🗺️ Breadcrumb Trails: ENABLED")
    print("  ⚡ Press Ctrl+C to stop")
    print("=" * 70)
    print()
    
    flask_app.run(host="0.0.0.0", port=5001, debug=False, threaded=True, use_reloader=False)
