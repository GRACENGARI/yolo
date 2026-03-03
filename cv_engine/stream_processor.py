import cv2
import time
import requests
import json
import logging
import numpy as np
import threading
import queue
import os
import argparse
from flask import Flask, Response, request

# Import configuration
try:
    from .config import config
    from .utils.auth import AuthManager
    from .utils.stream_manager import StreamManager
    from .utils.vram_monitor import VRAMMonitor
    from .core.deep_sort.tracker import Tracker as DeepSortTracker
    from .core.deep_sort import nn_matching
    from .core.deep_sort.detection import Detection
    from .core.face_recognition import FaceIdentifier
    from .core.forensic_enhancer_v2 import AdvancedForensicEnhancer
    from .core.reid_encoder import OSNetEncoder
except (ImportError, ValueError):
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from config import config
    from utils.auth import AuthManager
    from utils.stream_manager import StreamManager
    from utils.vram_monitor import VRAMMonitor
    from core.deep_sort.tracker import Tracker as DeepSortTracker
    from core.deep_sort import nn_matching
    from core.deep_sort.detection import Detection
    from core.face_recognition import FaceIdentifier
    from core.forensic_enhancer_v2 import AdvancedForensicEnhancer
    from core.reid_encoder import OSNetEncoder

# MJPEG Server Setup
flask_app = Flask(__name__)
output_frame = None
lock = threading.Lock()

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)

class SightingReporter:
    def __init__(self, backend_url, auth_manager):
        self.backend_url = backend_url
        self.auth = auth_manager
        self.queue = queue.Queue(maxsize=config.REPORT_QUEUE_SIZE)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        self.logger = logging.getLogger("SIGHTING_REPORTER")

    def report(self, data):
        try:
            self.queue.put_nowait(data)
        except queue.Full:
            self.logger.warning("Reporting queue full, dropping sighting.")

    def _worker(self):
        while not self.stop_event.is_set():
            try:
                data = self.queue.get(timeout=1)
                try:
                    res = self.auth.make_request('POST', f"{self.backend_url}sightings/", json=data)
                    if res and res.status_code != 201:
                        self.logger.error(f"Failed to report sighting: {res.status_code}")
                except Exception as e:
                    self.logger.error(f"Network error reporting sighting: {e}")
                finally:
                    self.queue.task_done()
            except queue.Empty:
                continue

class SimpleEncoder:
    def __call__(self, frame, boxes):
        features = []
        for box in boxes:
            x, y, w, h = map(int, box)
            x, y = max(0, x), max(0, y)
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                features.append(np.random.rand(128))
                continue
            avg_color = cv2.resize(roi, (8, 8)).flatten()
            feature = avg_color / 255.0 
            features.append(feature)
        return np.array(features)

class StreamProcessor:
    def __init__(self, source="people.mp4", backend_url=DEFAULT_BACKEND, detection_threshold=0.4, device=None):
        self.source = source
        self.backend_url = backend_url
        self.detection_threshold = detection_threshold
        self.device = device or os.getenv("DEVICE", "cpu")
        
        self.logger = logging.getLogger("CV_ENGINE")
        logging.basicConfig(level=logging.INFO)
        
        self.logger.info(f"Initializing YOLOv8n (Detection) on {self.device}...")
        self.model = YOLO("yolov8n.pt")
        
        try:
            if self.device in ['cuda', 'gpu']:
                self.model.to('cuda')
                self.model.half() # FP16 for GPU
                self.logger.info("YOLOv8n running on CUDA (FP16)")
            else:
                self.model.to('cpu')
                self.logger.info("YOLOv8n running on CPU (FP32)")
        except Exception as e:
            self.logger.warning(f"Failed to set device {self.device}: {e}. Falling back to CPU.")
            self.model.to('cpu')
        
        self.logger.info(f"Initializing ArcFace (Identity) on {self.device}...")
        self.face_id = FaceIdentifier(device=self.device)

        self.logger.info(f"Initializing Forensic Enhancer on {self.device}...")
        self.enhancer = ForensicEnhancer(device=self.device)
        
        self.logger.info("Initializing DeepSORT Tracker...")
        max_cosine_distance = 0.4
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, None)
        self.tracker = DeepSortTracker(metric)
        self.encoder = SimpleEncoder()
        
        # Sighting Reporter (Async/Concurrency Management)
        self.reporter = SightingReporter(self.backend_url)
        
        # Priority Targets (for Latency Optimization)
        self.priority_targets = [] # List of names or plates to alert on immediately
        
        # Track-based Reporting Control: { (camera_id, track_id): last_report_time }
        self.reported_tracks = {}
        self.report_interval = 30 # Re-report every 30 seconds if still in view
        
        # Metadata storage for tracks: { track_id: { 'class_id': int, 'identity': str } }
        self.track_metadata = {}

        self.camera_id = self._setup_camera()
        self.is_running = True

    def set_priority_target(self, target):
        self.logger.info(f"Priority Target Set: {target}")
        if target not in self.priority_targets:
            self.priority_targets.append(target)

    def _get_camera_coords(self):
        """Helper to get current camera lat/lng from backend or local cache."""
        try:
            res = requests.get(f"{self.backend_url}cameras/{self.camera_id}/", timeout=1)
            if res.status_code == 200:
                data = res.json()
                return data.get('latitude'), data.get('longitude')
        except:
            pass
        return -1.285, 36.821 # Default fallback

    def _setup_camera(self):
        device_id = "CAM-LIVE-SIM"
        try:
            get_res = requests.get(f"{self.backend_url}cameras/", timeout=2)
            if get_res.status_code == 200:
                cameras = get_res.json()
                for cam in cameras:
                    if cam['device_id'] == device_id:
                        return cam['id']

            res = requests.post(f"{self.backend_url}cameras/", json={
                "device_id": device_id,
                "location_name": "Grid Sector Alpha-01",
                "latitude": -1.285,
                "longitude": 36.821,
                "status": "ONLINE"
            }, timeout=2)
            if res.status_code == 201:
                return res.json().get('id')
        except:
            self.logger.error("Failed to connect to backend during camera setup. Fail-soft mode active.")
            return None

    def classify_color(self, img_roi):
        if img_roi.size == 0: return "UNKNOWN"
        h, w, _ = img_roi.shape
        center = img_roi[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
        if center.size == 0: return "UNKNOWN"
        b, g, r = np.mean(center, axis=(0, 1))
        
        if r > 150 and g < 100 and b < 100: return "RED"
        if b > 150 and r < 100 and g < 120: return "BLUE"
        if g > 150 and r < 100 and b < 100: return "GREEN"
        if r > 200 and g > 200 and b > 200: return "WHITE"
        if r < 50 and g < 50 and b < 50: return "BLACK"
        return "GREY"

    def process_stream(self):
        global output_frame, lock
        self.logger.info(f"Starting Video Capture: {self.source}")
        
        while self.is_running:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                self.logger.error(f"Failed to open source {self.source}. Retrying in 5s...")
                time.sleep(5)
                continue

            # FPS calculation for Success Criteria monitoring
            fps_start_time = time.time()
            fps_counter = 0
            
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    if "rtsp" in str(self.source).lower():
                        self.logger.warning("RTSP Stream disconnected. Attempting reconnection...")
                        break # Break inner loop to trigger cap.release() and reconnect
                    else:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop local video
                        continue

                # 1. DETECTION (YOLO)
                results = self.model(frame, verbose=False, stream=False, device=self.device)
                detections = []
                
                # COCO: 0=Person, 2=Car, 3=Motorcycle, 5=Bus, 7=Truck
                target_classes = [0, 2, 3, 5, 7]
                
                raw_dets = [] # Keep raw for association
                
                for result in results:
                    for r in result.boxes.data.tolist():
                        x1, y1, x2, y2, score, cls = r
                        cls = int(cls)
                        if cls in target_classes and score > self.detection_threshold:
                            detections.append([int(x1), int(y1), int(x2), int(y2), score])
                            raw_dets.append({'bbox': [x1, y1, x2, y2], 'class': cls})

                # 2. TRACKING (DeepSORT)
                self.update_tracker(frame, detections, raw_dets)

                # 3. ANALYSIS & VISUALIZATION
                for track in self.tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    
                    track_id = track.track_id
                    bbox = track.to_tlbr()
                    meta = self.track_metadata.get(track_id, {'class_id': 0, 'identity': 'Unknown'})
                    class_id = meta.get('class_id', 0)
                    
                    # Default Visuals
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # --- LOGIC SPLIT ---
                    
                    # A. PERSON PIPELINE (ArcFace + Enhancement)
                    if class_id == 0:
                        box_color = (0, 255, 65) # Green
                        embedding = None
                        
                        # Try to ID or at least get embedding
                        if track.time_since_update == 0:
                            face_roi = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                            
                            # Register first person as 'Juma' for demo
                            if track_id == 1 and 'Juma Macharia' not in self.face_id.known_faces:
                                self.face_id.register_face(face_roi, "Juma Macharia")
                            
                            name, conf, embedding = self.face_id.identify(face_roi)
                            
                            # TRIGGER LOGIC: Enhance if distorted or low confidence
                            if (conf < 0.4 or face_roi.shape[0] < 80) and face_roi.size > 0:
                                enhanced_face = self.enhancer.enhance_face(face_roi)
                                name_e, conf_e, embedding_e = self.face_id.identify(enhanced_face)
                                
                                if conf_e > conf:
                                    self.logger.info(f"Forensic Enhancement Improved Confidence: {conf:.2f} -> {conf_e:.2f}")
                                    name, conf, embedding = name_e, conf_e, embedding_e
                                    # Log Audit Comparison
                                    self.enhancer.save_forensic_audit(face_roi, enhanced_face, track_id)

                            if name != "Unknown" and meta['identity'] == 'Unknown':
                                meta['identity'] = name
                                self.track_metadata[track_id] = meta
                        
                        identity = meta['identity']
                        label = f"PERSON // {identity}"
                        if identity != "Unknown":
                            box_color = (0, 0, 255) # Red for known targets
                            label = f"TARGET: {identity}"
                        
                        self.report_sighting(track, "N/A", "N/A", identity, is_person=True, embedding=embedding)

                    # B. VEHICLE PIPELINE (YOLO+Attr)
                    else:
                        box_color = (255, 200, 0) # Cyan/Orange
                        veh_roi = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                        color = self.classify_color(veh_roi)
                        
                        veh_types = {2: 'CAR', 3: 'MOTO', 5: 'BUS', 7: 'TRUCK'}
                        v_type = veh_types.get(class_id, 'VEHICLE')
                        
                        label = f"{v_type} // {color}"
                        self.report_sighting(track, f"Unknown-{track_id}", color, v_type, is_person=False, embedding=None)

                    # Draw
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 1)
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

                # Success Criteria Monitoring: FPS
                fps_counter += 1
                if time.time() - fps_start_time > 1:
                    self.logger.info(f"Performance: {fps_counter} FPS")
                    fps_counter = 0
                    fps_start_time = time.time()

                # Overlay
                cv2.putText(frame, f"HAWKEYE v2.1 // ORCHESTRATION ENABLED", (20, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 65), 2)

                with lock:
                    output_frame = frame.copy()

    def update_tracker(self, frame, detections, raw_dets):
        if len(detections) == 0:
            self.tracker.predict()
            self.tracker.update([])
            return
            
        bboxes = np.asarray([d[:4] for d in detections])
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2] # to xywh
        scores = [d[4] for d in detections]
        
        features = self.encoder(frame, bboxes)
        dets = [Detection(bbox, score, feature) for bbox, score, feature in zip(bboxes, scores, features)]
        
        self.tracker.predict()
        self.tracker.update(dets)
        
        # ASSOCIATE CLASSES TO TRACKS
        for track in self.tracker.tracks:
            if not track.is_confirmed(): continue
            
            # If we don't know the class, find the closest detection
            if track.track_id not in self.track_metadata:
                t_bbox = track.to_tlbr()
                best_iou = 0
                best_cls = 0
                
                # Simple IoU check against raw_dets
                t_area = (t_bbox[2]-t_bbox[0]) * (t_bbox[3]-t_bbox[1])
                for rd in raw_dets:
                    r_bbox = rd['bbox']
                    # Intersect
                    xA = max(t_bbox[0], r_bbox[0])
                    yA = max(t_bbox[1], r_bbox[1])
                    xB = min(t_bbox[2], r_bbox[2])
                    yB = min(t_bbox[3], r_bbox[3])
                    interArea = max(0, xB - xA) * max(0, yB - yA)
                    
                    if interArea > 0:
                        iou = interArea / float(t_area + ((r_bbox[2]-r_bbox[0])*(r_bbox[3]-r_bbox[1])) - interArea)
                        if iou > best_iou:
                            best_iou = iou
                            best_cls = rd['class']
                
                if best_iou > 0.3:
                    self.track_metadata[track.track_id] = {'class_id': best_cls, 'identity': 'Unknown'}

    def report_sighting(self, track, plate, color, make, is_person=False, embedding=None):
        now = time.time()
        track_key = (self.camera_id, track.track_id)
        
        # Check if we should report this track again
        is_priority = (plate in self.priority_targets or make in self.priority_targets)
        last_report = self.reported_tracks.get(track_key, 0)
        
        # Report if: Never reported OR interval passed OR it's a priority target we just identified
        if now - last_report < self.report_interval and not is_priority:
            return
            
        self.reported_tracks[track_key] = now
        
        bbox = track.to_tlbr()
        # Use camera coordinates for "Road Snapping" (eliminating jitter)
        lat, lng = self._get_camera_coords()
        
        data = {
            "camera_id": self.camera_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "confidence_score": 0.95,
            "bbox_data": {"x": int(bbox[0]), "y": int(bbox[1]), "w": int(bbox[2]-bbox[0]), "h": int(bbox[3]-bbox[1])},
            "snapshot_url": "http://localhost:5001/video_feed", 
            "detected_plate": plate,
            "detected_color": color,
            "detected_make": make,
            "is_person": is_person,
            "gps_lat": lat,
            "gps_lng": lng,
            "embedding": embedding.tolist() if embedding is not None else None
        }
        # Asynchronous reporting via queue
        self.reporter.report(data)

def generate():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None: continue
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
            if not flag: continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@flask_app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@flask_app.route("/briefing", methods=['POST'])
def update_briefing():
    data = request.json
    target = data.get('target')
    if target:
        processor.set_priority_target(target)
        return {"status": "Target Acquired"}, 200
    return {"status": "No Target Specified"}, 400

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MWEWE CV Engine // Phase 4")
    parser.add_argument("--source", type=str, default="people.mp4", help="Video source (file path or RTSP URL)")
    parser.add_argument("--backend", type=str, default=DEFAULT_BACKEND, help="Backend API URL")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "gpu"], help="Hardware device to use")
    parser.add_argument("--threshold", type=float, default=0.4, help="Detection threshold")
    
    args = parser.parse_args()

    # Normalization: map 'gpu' to 'cuda' for YOLO/ArcFace
    target_device = 'cuda' if args.device == 'gpu' else args.device

    processor = StreamProcessor(
        source=args.source, 
        backend_url=args.backend, 
        device=target_device,
        detection_threshold=args.threshold
    )
    
    t = threading.Thread(target=processor.process_stream)
    t.daemon = True
    t.start()
    
    # Concurrency Management: Run Flask in threaded mode
    flask_app.run(host="0.0.0.0", port=5001, debug=False, threaded=True, use_reloader=False)
