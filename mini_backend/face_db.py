"""
Face Database Manager
Handles face registration, identification, and tracking
"""
import numpy as np
import cv2
import logging
from datetime import datetime

logger = logging.getLogger("MINI_BACKEND.FACE_DB")

class FaceDatabase:
    """Manages face recognition without InsightFace dependency"""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.face_cascade = None
        
        # Try to load OpenCV face detector
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            logger.info("✓ OpenCVl_persons.items():
            person_data = self.vector_store.get_person(person_id)
            if person_data:
                self.known_faces[data['name']] = person_data['embedding']
    
    def register_face(self, embedding, name):
        """Register a new face"""
        if name in self.known_faces:
            logger.warning(f"Face already registered: {name}")
            return False
        
        person_id = self.vector_store.register_person(name, embedding)
        self.known_faces[name] = np.array(embedding)
        logger.info(f"✓ Registered new face: {name} (ID: {person_id})")
        return person_id
    
    def identify(self, embedding, threshold=0.4):
        """
        Identify a face from its embedding
        Returns: (person_id, name, confidence) or (None, "Unknown", 0.0)
        """
        results = self.vector_store.search_by_embedding(embedding, threshold=threshold, top_k=1)
        
        if results:
            person_id, name, confidence = results[0]
            return person_id, name, confidence
        
        return None, "Unknown", 0.0
    
    def add_sighting(self, person_id, location, timestamp, confidence, bbox=None):
        """Record a sighting"""
        return self.vector_store.add_sighting(person_id, location, timestamp, confidence, bbox)
    
    def forensic_search(self, query_embedding):
        """Search for all sightings matching an embedding"""
        return self.vector_store.forensic_search(query_embedding)
    
    def get_breadcrumb_trail(self, person_id):
        """Get movement trail for a person"""
        return self.vector_store.get_breadcrumb_trail(person_id)
    
    def get_all_known_faces(self):
        """Get list of all known faces"""
        return list(self.known_faces.keys())
    
    def get_statistics(self):
        """Get database statistics"""
        all_persons = self.vector_store.get_all_persons()
        total_sightings = sum(p['sighting_count'] for p in all_persons.values())
        
        return {
            'total_persons': len(all_persons),
            'total_sightings': total_sightings,
            'known_faces': list(all_persons.keys())
        }
