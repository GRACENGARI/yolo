"""
HAWKEYE CV-Engine Configuration Manager
Centralized configuration with environment variable support
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)

class Config:
    """Centralized configuration for CV-Engine"""
    
    # Hardware & Performance
    DEVICE = os.getenv('DEVICE', 'cpu')
    DETECTION_THRESHOLD = float(os.getenv('DETECTION_THRESHOLD', '0.4'))
    VRAM_LIMIT_MB = int(os.getenv('VRAM_LIMIT_MB', '4096'))
    ENABLE_FP16 = os.getenv('ENABLE_FP16', 'true').lower() == 'true'
    
    # Video Source
    VIDEO_SOURCE = os.getenv('VIDEO_SOURCE', 'people.mp4')
    RTSP_RECONNECT_ATTEMPTS = int(os.getenv('RTSP_RECONNECT_ATTEMPTS', '5'))
    RTSP_RECONNECT_DELAY = int(os.getenv('RTSP_RECONNECT_DELAY', '3'))
    
    # Backend Communication
    BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:8000/api/v1/')
    API_KEY = os.getenv('API_KEY', '')
    ENABLE_MTLS = os.getenv('ENABLE_MTLS', 'false').lower() == 'true'
    MTLS_CERT_PATH = os.getenv('MTLS_CERT_PATH', './certs/client.crt')
    MTLS_KEY_PATH = os.getenv('MTLS_KEY_PATH', './certs/client.key')
    MTLS_CA_PATH = os.getenv('MTLS_CA_PATH', './certs/ca.crt')
    
    # Camera Configuration
    CAMERA_ID = os.getenv('CAMERA_ID', 'CAM-LIVE-SIM')
    CAMERA_LOCATION = os.getenv('CAMERA_LOCATION', 'Grid Sector Alpha-01')
    CAMERA_LAT = float(os.getenv('CAMERA_LAT', '-1.285'))
    CAMERA_LNG = float(os.getenv('CAMERA_LNG', '36.821'))
    
    # AI Models
    YOLO_MODEL = os.getenv('YOLO_MODEL', 'yolov8n.pt')
    ARCFACE_MODEL = os.getenv('ARCFACE_MODEL', 'buffalo_s')
    REID_MODEL = os.getenv('REID_MODEL', 'osnet_x1_0')
    ENHANCEMENT_MODEL = os.getenv('ENHANCEMENT_MODEL', 'gfpgan')
    
    # Forensic Enhancement
    ENABLE_ENHANCEMENT = os.getenv('ENABLE_ENHANCEMENT', 'true').lower() == 'true'
    ENHANCEMENT_THRESHOLD = float(os.getenv('ENHANCEMENT_THRESHOLD', '0.4'))
    MIN_FACE_SIZE = int(os.getenv('MIN_FACE_SIZE', '20'))
    SAVE_FORENSIC_AUDIT = os.getenv('SAVE_FORENSIC_AUDIT', 'true').lower() == 'true'
    FORENSIC_AUDIT_DIR = os.getenv('FORENSIC_AUDIT_DIR', './forensic_audit')
    
    # Tracking & Reporting
    REPORT_INTERVAL = int(os.getenv('REPORT_INTERVAL', '30'))
    MAX_COSINE_DISTANCE = float(os.getenv('MAX_COSINE_DISTANCE', '0.4'))
    ENABLE_REID = os.getenv('ENABLE_REID', 'true').lower() == 'true'
    
    # Flask Server
    FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    FLASK_PORT = int(os.getenv('FLASK_PORT', '5001'))
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'cv_out.log')
    
    @classmethod
    def validate(cls):
        """Validate critical configuration"""
        errors = []
        
        if cls.DEVICE not in ['cpu', 'cuda', 'gpu']:
            errors.append(f"Invalid DEVICE: {cls.DEVICE}. Must be cpu, cuda, or gpu")
        
        if not 0 <= cls.DETECTION_THRESHOLD <= 1:
            errors.append(f"DETECTION_THRESHOLD must be between 0 and 1")
        
        if cls.ENABLE_MTLS:
            for path_attr in ['MTLS_CERT_PATH', 'MTLS_KEY_PATH', 'MTLS_CA_PATH']:
                path = getattr(cls, path_attr)
                if not Path(path).exists():
                    errors.append(f"mTLS enabled but {path_attr} not found: {path}")
        
        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(errors))
        
        return True
    
    @classmethod
    def summary(cls):
        """Return configuration summary for logging"""
        return f"""
╔══════════════════════════════════════════════════════════════╗
║           HAWKEYE CV-Engine Configuration                    ║
╠══════════════════════════════════════════════════════════════╣
║ Device: {cls.DEVICE:<50} ║
║ Video Source: {cls.VIDEO_SOURCE:<44} ║
║ Backend: {cls.BACKEND_URL:<47} ║
║ Enhancement: {('Enabled' if cls.ENABLE_ENHANCEMENT else 'Disabled'):<45} ║
║ ReID Model: {('Advanced' if cls.ENABLE_REID else 'Basic'):<46} ║
║ mTLS: {('Enabled' if cls.ENABLE_MTLS else 'Disabled'):<50} ║
╚══════════════════════════════════════════════════════════════╝
"""

# Create a singleton instance for easy importing
config = Config()
