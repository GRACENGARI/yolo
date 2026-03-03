"""
Mini Backend Server
Provides face recognition and forensic search APIs
"""
from flask import Flask, request, jsonify
import logging
from .simple_face_db import SimpleFaceDB
from .sighting_tracker import SightingTracker

logger = logging.getLogger("MINI_BACKEND.SERVER")

class MiniBackend:
    """
    Mini backend service for face recognition and tracking
    """
    
    def __init__(self):
        self.face_db = SimpleFaceDB()
        self.tracker = SightingTracker()
        
        logger.info("Mini Backend initialized")
    
    def register_face(self, image, name):
        """Register a new face"""
        return self.face_db.register_face(image, name)
    
    def identify_face(self, image):
        """Identify a face in image"""
        name, confidence, face_crop = self.face_db.identify(image)
        return name, confidence, face_crop
    
    def add_sighting(self, person_name, confidence, camera_id="CAM-01", 
                    location="Unknown", bbox=None):
        """Add a sighting to tracker"""
        return self.tracker.add_sighting(
            person_name, confidence, camera_id, location, bbox
        )
    
    def get_breadcrumb_trail(self, person_name):
        """Get breadcrumb trail for person"""
        return self.tracker.get_breadcrumb_trail(person_name)
    
    def forensic_search(self, person_name=None, camera_id=None):
        """Perform forensic search"""
        return self.tracker.forensic_search(person_name, camera_id=camera_id)
    
    def get_known_faces(self):
        """Get list of known faces"""
        return self.face_db.get_all_faces()
    
    def get_statistics(self):
        """Get tracking statistics"""
        return self.tracker.get_statistics()

# Global instance
mini_backend = None

def get_mini_backend():
    """Get or create mini backend instance"""
    global mini_backend
    if mini_backend is None:
        mini_backend = MiniBackend()
    return mini_backend
