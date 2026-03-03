"""
HAWKEYE Vehicle Tracker - SIMPLIFIED & WORKING VERSION
Clean, functional vehicle tracking with working buttons
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

# Load environment
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VEHICLE_SIMPLE")

# Import predictive analytics
try:
    from predictive_analytics import PredictiveAnalytics
    analytics_engine = PredictiveAnalytics()
    ANALYTICS_ENABLED = True
    logger.info("✓ Predictive Analytics enabled")
except Exception as e:
    logger.warning(f"Predictive analytics not available: {e}")
    ANALYTICS_ENABLED = False

flask_app = Flask(__name__)
output_frame = None
lock = threading.Lock()

DEVICE = os.getenv('DEVICE', 'cpu')
VIDEO_SOURCE = os.getenv('VIDEO_SOURCE', 'surveillance_video.mp4')
DETECTION_THRESHOLD = 0.5

print("=" * 70)
print("  HAWKEYE VEHICLE TRACKER - SIMPLIFIED")
print("=" * 70)

model = YOLO("yolov8n.pt")
model.to(DEVICE)
tracker = SightingTracker()

vehicle_tracks = {}
next_vehicle_id = 1
target_vehicles = {}
last_sighting_time = {}

def get_vehicle_info(track_id):
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
    
    vehicle_tracks[track_id]['last_seen'] = time.time()
    vehicle_tracks[track_id]['sightings'] += 1
    
    return vehicle_tracks[track_id]

def register_vehicle_target(track_id, description):
    target_vehicles[track_id] = description
    if track_id in vehicle_tracks:
        vehicle_tracks[track_id]['description'] = description
        vehicle_tracks[track_id]['is_target'] = True
    logger.info(f"✓ Registered: {description} (ID: {track_id})")

def process_video():
    global output_frame
    
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    fps_start = time.time()
    fps_counter = 0
    
    from collections import defaultdict
    track_history = defaultdict(lambda: [])
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        results = model.track(frame, persist=True, verbose=False, device=DEVICE, classes=[2, 3, 5, 7])
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, track_id, conf, cls in zip(boxes, track_ids, confidences, classes):
                if conf > DETECTION_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box)
                    
                    vehicle_info = get_vehicle_info(track_id)
                    vehicle_desc = vehicle_info['description']
                    is_target = vehicle_info['is_target']
                    
                    current_time = time.time()
                    last_time = last_sighting_time.get(vehicle_desc, 0)
                    if current_time - last_time > 2:
                        tracker.add_sighting(
                            person_name=vehicle_desc,
                            confidence=float(conf),
                            camera_id="CAM-01",
                            location="Sector Alpha",
                            bbox=(x1, y1, x2-x1, y2-y1)
                        )
                        last_sighting_time[vehicle_desc] = current_time
                    
                    veh_types = {2: 'CAR', 3: 'MOTO', 5: 'BUS', 7: 'TRUCK'}
                    veh_type = veh_types.get(cls, 'VEH')
                    
                    if is_target:
                        color = (0, 0, 255)
                        label = f"TARGET: {vehicle_desc}"
                        thickness = 3
                        
                        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(frame, (x1, y1-label_h-10), (x1+label_w+10, y1), color, -1)
                        cv2.putText(frame, label, (x1+5, y1-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        track_history[track_id].append((int((x1+x2)/2), int((y1+y2)/2)))
                        if len(track_history[track_id]) > 30:
                            track_history[track_id].pop(0)
                        
                        points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
                        if len(points) > 1:
                            cv2.polylines(frame, [points], False, (0, 0, 255), 3)
                    else:
                        color = (0, 165, 255)
                        label = f"{veh_type} ID:{track_id}"
                        thickness = 2
                        cv2.putText(frame, label, (x1, y1-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        fps_counter += 1
        if time.time() - fps_start > 1:
            logger.info(f"{fps_counter} FPS | {len(vehicle_tracks)} vehicles | {len([v for v in vehicle_tracks.values() if v['is_target']])} targets")
            fps_counter = 0
            fps_start = time.time()
        
        cv2.putText(frame, "HAWKEYE VEHICLE TRACKER", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 65), 2)
        
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
    target_list = [v['description'] for v in vehicle_tracks.values() if v['is_target']]
    
    # Get recent sightings for analytics
    recent_sightings = tracker.get_recent_sightings(20)
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>HAWKEYE Vehicle Tracker</title>
        <style>
            body { background: #000; color: #0f0; font-family: 'Courier New', monospace; margin: 0; padding: 20px; }
            h1 { color: #0f0; border-bottom: 2px solid #0f0; }
            .video { width: 100%; border: 3px solid #0f0; margin: 20px 0; }
            .panel { background: #111; border: 1px solid #0f0; padding: 20px; margin: 15px 0; }
            input { background: #222; color: #0f0; border: 2px solid #0f0; padding: 12px; 
                    font-family: 'Courier New', monospace; font-size: 1em; margin: 5px; }
            button { background: #0f0; color: #000; border: none; padding: 12px 24px; 
                     font-family: 'Courier New', monospace; font-weight: bold; cursor: pointer; margin: 5px; }
            button:hover { background: #0ff; }
            .target-btn { background: #f00; color: #fff; font-size: 1.1em; padding: 15px 30px; }
            .target-btn:hover { background: #ff4444; }
            .result { background: #1a1a1a; padding: 15px; margin: 10px 0; border-left: 4px solid #0f0; }
            .target { color: #f00; font-size: 1.2em; font-weight: bold; }
            .stat-box { background: #1a1a1a; border: 2px solid #0ff; padding: 15px; margin: 10px; 
                        display: inline-block; min-width: 200px; text-align: center; }
            .stat-number { font-size: 2.5em; color: #0ff; font-weight: bold; }
            .stat-label { font-size: 1em; color: #0f0; margin-top: 5px; }
            .sighting-item { background: #0a0a0a; padding: 10px; margin: 5px 0; border-left: 3px solid #0f0; }
            .analytics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; }
        </style>
    </head>
    <body>
        <h1>🚗 HAWKEYE VEHICLE TRACKER</h1>
        
        <img src="/video_feed" class="video" />
        
        <div class="panel">
            <h2 style="color: #ff0;">🎯 Registered Targets</h2>
            {% if target_list %}
                {% for target in target_list %}
                <div class="target">✓ {{ target }}</div>
                {% endfor %}
            {% else %}
                <p style="color: #ff0;">No targets registered yet</p>
            {% endif %}
        </div>
        
        <div class="panel">
            <h2 style="color: #ff0;">➕ Register Vehicle as Target</h2>
            <p style="color: #0ff;">Watch video → Note Track ID → Register below</p>
            <input type="number" id="track_id" placeholder="Track ID (e.g., 15)" style="width: 200px;" />
            <input type="text" id="description" placeholder="Description (e.g., White Toyota)" style="width: 400px;" />
            <button class="target-btn" onclick="register()">REGISTER TARGET</button>
        </div>
        
        <div class="panel">
            <h2 style="color: #ff0;">📊 ANALYTICS DASHBOARD</h2>
            
            <div class="analytics-grid">
                <div class="stat-box">
                    <div class="stat-number">{{ stats.total_sightings }}</div>
                    <div class="stat-label">TOTAL SIGHTINGS</div>
                </div>
                
                <div class="stat-box">
                    <div class="stat-number">{{ stats.by_person|length }}</div>
                    <div class="stat-label">UNIQUE VEHICLES</div>
                </div>
                
                <div class="stat-box">
                    <div class="stat-number">{{ target_list|length }}</div>
                    <div class="stat-label">TARGET VEHICLES</div>
                </div>
                
                <div class="stat-box">
                    <div class="stat-number">{{ recent_sightings|length }}</div>
                    <div class="stat-label">RECENT ACTIVITY</div>
                </div>
            </div>
            
            {% if target_list %}
            <h3 style="color: #f00; margin-top: 30px;">🎯 TARGET VEHICLE ACTIVITY</h3>
            {% for target in target_list %}
                {% for v in stats.by_person %}
                    {% if v.person_name == target %}
                    <div class="result" style="border-left: 4px solid #f00; background: #2a0000;">
                        <strong style="color: #f00; font-size: 1.3em;">{{ v.person_name }}</strong>
                        <div style="margin-top: 10px;">
                            <span style="color: #0ff; font-size: 1.2em;">{{ v.count }} sightings</span>
                            <button onclick="searchVehicle('{{ v.person_name }}')" style="margin-left: 20px;">🔍 Forensic Search</button>
                            <button onclick="getTrail('{{ v.person_name }}')">🗺️ Movement Trail</button>
                        </div>
                    </div>
                    {% endif %}
                {% endfor %}
            {% endfor %}
            {% endif %}
            
            <h3 style="margin-top: 30px; color: #0ff;">📈 TOP 10 MOST ACTIVE VEHICLES</h3>
            {% for v in stats.by_person[:10] %}
            <div class="result">
                <strong style="font-size: 1.1em;">{{ v.person_name }}</strong>: 
                <span style="color: #0ff;">{{ v.count }} sightings</span>
                <button onclick="searchVehicle('{{ v.person_name }}')" style="margin-left: 15px; padding: 8px 16px;">🔍 Search</button>
                <button onclick="getTrail('{{ v.person_name }}')" style="padding: 8px 16px;">🗺️ Trail</button>
                <button onclick="predictVehicle('{{ v.person_name }}')" style="padding: 8px 16px; background: #ff0; color: #000;">🔮 Predict</button>
                <button onclick="getRisk('{{ v.person_name }}')" style="padding: 8px 16px; background: #f80; color: #000;">⚠️ Risk</button>
            </div>
            {% endfor %}
            
            <h3 style="margin-top: 30px; color: #ff0;">🕐 RECENT SIGHTINGS (Last 20)</h3>
            {% for s in recent_sightings %}
            <div class="sighting-item">
                <div style="color: #0ff; font-size: 1.1em;">{{ s.person_name }}</div>
                <div style="color: #0f0; font-size: 0.9em;">
                    🕐 {{ s.timestamp }} | 📹 {{ s.camera_id }} | 📍 {{ s.location }} | 
                    Confidence: {{ (s.confidence * 100)|int }}%
                </div>
            </div>
            {% endfor %}
        </div>
        
        <div class="panel" id="results" style="display: none;">
            <h2 style="color: #0ff;">📋 FORENSIC RESULTS</h2>
            <div id="results_content"></div>
        </div>
        
        <script>
            // Auto-refresh analytics every 5 seconds
            setInterval(() => {
                if (document.getElementById('results').style.display === 'none') {
                    location.reload();
                }
            }, 5000);
            
            function register() {
                const id = document.getElementById('track_id').value;
                const desc = document.getElementById('description').value;
                
                if (!id || !desc) {
                    alert('⚠️ Enter both Track ID and Description');
                    return;
                }
                
                fetch('/api/register', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({track_id: parseInt(id), description: desc})
                })
                .then(r => r.json())
                .then(data => {
                    alert('✓ ' + data.message);
                    location.reload();
                })
                .catch(e => alert('❌ Error: ' + e));
            }
            
            function searchVehicle(name) {
                document.getElementById('results').style.display = 'block';
                fetch('/api/forensic_search?person=' + encodeURIComponent(name))
                    .then(r => r.json())
                    .then(data => {
                        let html = '<h3 style="color: #f00;">🔍 FORENSIC SEARCH: ' + name + '</h3>';
                        html += '<p style="color: #ff0; font-size: 1.2em;">Found <strong>' + data.results.length + '</strong> sightings</p>';
                        html += '<button onclick="document.getElementById(\'results\').style.display=\'none\'" style="background: #f00;">✖ Close</button>';
                        
                        const recent = data.results.slice(0, 20);
                        recent.forEach((s, i) => {
                            html += '<div class="result">';
                            html += '<div style="color: #0ff; font-size: 1.1em;">#' + (i+1) + ' - ' + s.timestamp + '</div>';
                            html += '<div style="margin-top: 5px;">📹 ' + s.camera_id + ' | 📍 ' + s.location + ' | Confidence: ' + (s.confidence*100).toFixed(0) + '%</div>';
                            html += '</div>';
                        });
                        
                        document.getElementById('results_content').innerHTML = html;
                        document.getElementById('results').scrollIntoView({behavior: 'smooth'});
                    });
            }
            
            function getTrail(name) {
                document.getElementById('results').style.display = 'block';
                fetch('/api/breadcrumb?person=' + encodeURIComponent(name))
                    .then(r => r.json())
                    .then(data => {
                        let html = '<h3 style="color: #f00;">🗺️ MOVEMENT TRAIL: ' + name + '</h3>';
                        html += '<p style="color: #ff0; font-size: 1.2em;">' + data.trail.length + ' locations tracked</p>';
                        html += '<button onclick="document.getElementById(\'results\').style.display=\'none\'" style="background: #f00;">✖ Close</button>';
                        
                        data.trail.slice(-25).reverse().forEach((s, i) => {
                            html += '<div class="result">';
                            html += '<div style="color: #f00; font-size: 1.1em;">📍 Location #' + (i+1) + '</div>';
                            html += '<div style="margin-top: 5px;">🕐 ' + s.timestamp + '</div>';
                            html += '<div>📹 ' + s.camera_id + ' | 📍 ' + s.location + '</div>';
                            html += '</div>';
                        });
                        
                        document.getElementById('results_content').innerHTML = html;
                        document.getElementById('results').scrollIntoView({behavior: 'smooth'});
                    });
            }
            
            function predictVehicle(name) {
                document.getElementById('results').style.display = 'block';
                fetch('/api/analytics/predict?vehicle=' + encodeURIComponent(name))
                    .then(r => r.json())
                    .then(data => {
                        if (data.error) {
                            alert('⚠️ ' + data.error);
                            return;
                        }
                        
                        let html = '<h3 style="color: #ff0;">🔮 PREDICTIVE ANALYSIS: ' + name + '</h3>';
                        html += '<button onclick="document.getElementById(\'results\').style.display=\'none\'" style="background: #f00;">✖ Close</button>';
                        
                        html += '<div class="result" style="border-left: 4px solid #ff0; background: #2a2a00;">';
                        html += '<h4 style="color: #ff0;">NEXT APPEARANCE PREDICTION</h4>';
                        html += '<div style="font-size: 1.2em; margin: 10px 0;">';
                        html += '<div>⏰ Predicted Time: <span style="color: #0ff;">' + data.predicted_hour + ':00</span></div>';
                        html += '<div>📍 Likely Location: <span style="color: #0ff;">' + data.predicted_location + '</span></div>';
                        html += '<div>📹 Likely Camera: <span style="color: #0ff;">' + data.predicted_camera + '</span></div>';
                        html += '<div>🎯 Confidence: <span style="color: #0ff;">' + data.confidence + '</span></div>';
                        html += '</div></div>';
                        
                        document.getElementById('results_content').innerHTML = html;
                        document.getElementById('results').scrollIntoView({behavior: 'smooth'});
                    })
                    .catch(e => alert('❌ Error: ' + e));
            }
            
            function getRisk(name) {
                document.getElementById('results').style.display = 'block';
                fetch('/api/analytics/risk?vehicle=' + encodeURIComponent(name))
                    .then(r => r.json())
                    .then(data => {
                        if (data.error) {
                            alert('⚠️ ' + data.error);
                            return;
                        }
                        
                        let riskColor = data.risk_level === 'HIGH' ? '#f00' : data.risk_level === 'MEDIUM' ? '#f80' : '#0f0';
                        
                        let html = '<h3 style="color: ' + riskColor + ';">⚠️ RISK ASSESSMENT: ' + name + '</h3>';
                        html += '<button onclick="document.getElementById(\'results\').style.display=\'none\'" style="background: #f00;">✖ Close</button>';
                        
                        html += '<div class="result" style="border-left: 4px solid ' + riskColor + ';">';
                        html += '<h4 style="color: ' + riskColor + ';">RISK LEVEL: ' + data.risk_level + '</h4>';
                        html += '<div style="font-size: 1.5em; margin: 15px 0;">Risk Score: <span style="color: ' + riskColor + ';">' + data.risk_score + '/100</span></div>';
                        html += '<div style="margin: 10px 0;">Total Sightings: <span style="color: #0ff;">' + data.total_sightings + '</span></div>';
                        
                        if (data.risk_factors && data.risk_factors.length > 0) {
                            html += '<h4 style="color: #ff0; margin-top: 15px;">Risk Factors:</h4>';
                            data.risk_factors.forEach(factor => {
                                html += '<div style="margin: 5px 0; padding: 8px; background: #1a0000;">⚠️ ' + factor + '</div>';
                            });
                        } else {
                            html += '<div style="color: #0f0; margin-top: 10px;">✓ No significant risk factors detected</div>';
                        }
                        
                        html += '</div>';
                        
                        document.getElementById('results_content').innerHTML = html;
                        document.getElementById('results').scrollIntoView({behavior: 'smooth'});
                    })
                    .catch(e => alert('❌ Error: ' + e));
            }
        </script>
    </body>
    </html>
    """
    
    return render_template_string(html, target_list=target_list, stats=stats, recent_sightings=recent_sightings)

@flask_app.route("/api/register", methods=['POST'])
def api_register():
    try:
        data = request.json
        logger.info(f"Registration request received: {data}")
        
        track_id = data.get('track_id')
        description = data.get('description')
        
        if not track_id or not description:
            logger.error("Missing track_id or description")
            return jsonify({'success': False, 'message': 'Missing data'}), 400
        
        track_id = int(track_id)
        logger.info(f"Registering vehicle: ID={track_id}, Description={description}")
        
        register_vehicle_target(track_id, description)
        
        logger.info(f"✓ Successfully registered: {description} (ID: {track_id})")
        return jsonify({'success': True, 'message': f'Registered: {description}'})
    except Exception as e:
        logger.error(f"Registration error: {str(e)}", exc_info=True)
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

@flask_app.route("/api/analytics/predict")
def api_predict():
    if not ANALYTICS_ENABLED:
        return jsonify({'error': 'Analytics not available'}), 503
    
    vehicle = request.args.get('vehicle')
    if not vehicle:
        return jsonify({'error': 'Vehicle name required'}), 400
    
    prediction = analytics_engine.predict_next_appearance(vehicle)
    return jsonify(prediction if prediction else {'error': 'No data available'})

@flask_app.route("/api/analytics/risk")
def api_risk():
    if not ANALYTICS_ENABLED:
        return jsonify({'error': 'Analytics not available'}), 503
    
    vehicle = request.args.get('vehicle')
    if not vehicle:
        return jsonify({'error': 'Vehicle name required'}), 400
    
    risk = analytics_engine.get_vehicle_risk_score(vehicle)
    return jsonify(risk if risk else {'error': 'No data available'})

@flask_app.route("/api/analytics/hotspots")
def api_hotspots():
    if not ANALYTICS_ENABLED:
        return jsonify({'error': 'Analytics not available'}), 503
    
    hours = int(request.args.get('hours', 24))
    hotspots = analytics_engine.get_hotspot_analysis(hours)
    return jsonify({'hotspots': hotspots})

if __name__ == "__main__":
    video_thread = threading.Thread(target=process_video, daemon=True)
    video_thread.start()
    
    print()
    print("=" * 70)
    print("  🚀 RUNNING AT: http://localhost:5001")
    print("=" * 70)
    print()
    
    flask_app.run(host="0.0.0.0", port=5001, debug=False, threaded=True, use_reloader=False)
