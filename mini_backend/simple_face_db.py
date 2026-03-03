"""
Simple Face Database using OpenCV
Works without InsightFace - uses ORB features for face matching
"""
import cv2
import numpy as np
import pickle
import os
from pathlib import Path
import logging

logger = logging.getLogger("MINI_BACKEND.FACE_DB")

class SimpleFaceDB:
    """
    Simple face recognition using OpenCV ORB features
    No external dependencies required
    """
    
    def __init__(self, db_path="mini_backend/face_database.pkl"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # ORB feature detector
        self.orb = cv2.ORB_create(nfeatures=500)
        
        # BFMatcher for feature matching
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Database: {name: {'features': descriptors, 'image': face_image}}
        self.known_faces = {}
        
        # Load existing database
        self.load_database()
        
        logger.info(f"SimpleFaceDB initialized with {len(self.known_faces)} known faces")
    
    def detect_face(self, image):
        """Detect the largest face in image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None
        
        # Return largest face
        largest = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest
        return image[y:y+h, x:x+w]
    
    def extract_features(self, face_image):
        """Extract ORB features from face"""
        if face_image is None or face_image.size == 0:
            return None
        
        # Resize to standard size
        face_resized = cv2.resize(face_image, (128, 128))
        gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        return descriptors
    
    def register_face(self, image, name):
        """Register a new face"""
        face = self.detect_face(image)
        if face is None:
            logger.warning(f"No face detected for {name}")
            return False
        
        features = self.extract_features(face)
        if features is None:
            logger.warning(f"Could not extract features for {name}")
            return False
        
        self.known_faces[name] = {
            'features': features,
            'image': face,
            'registered_at': np.datetime64('now')
        }
        
        self.save_database()
        logger.info(f"✓ Registered: {name} ({len(features)} features)")
        return True
    
    def identify(self, image, threshold=30):
        """
        Identify a face in the image
        Returns: (name, confidence, face_crop)
        """
        face = self.detect_face(image)
        if face is None:
            return "Unknown", 0.0, None
        
        features = self.extract_features(face)
        if features is None:
            return "Unknown", 0.0, face
        
        best_name = "Unknown"
        best_score = 0
        
        for name, data in self.known_faces.items():
            known_features = data['features']
            
            try:
                # Match features
                matches = self.matcher.match(features, known_features)
                
                # Calculate match score (lower distance = better match)
                if len(matches) > 0:
                    avg_distance = sum(m.distance for m in matches) / len(matches)
                    # Convert to confidence (0-1)
                    confidence = max(0, 1 - (avg_distance / 100))
                    
                    if confidence > best_score and avg_distance < threshold:
                        best_score = confidence
                        best_name = name
            except Exception as e:
                logger.error(f"Error matching {name}: {e}")
                continue
        
        return best_name, best_score, face
    
    def save_database(self):
        """Save database to disk"""
        try:
            with open(self.db_path, 'wb') as f:
                pickle.dump(self.known_faces, f)
            logger.debug(f"Database saved: {len(self.known_faces)} faces")
        except Exception as e:
            logger.error(f"Failed to save database: {e}")
    
    def load_database(self):
        """Load database from disk"""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'rb') as f:
                    self.known_faces = pickle.load(f)
                logger.info(f"Database loaded: {len(self.known_faces)} faces")
            except Exception as e:
                logger.error(f"Failed to load database: {e}")
                self.known_faces = {}
        else:
            self.known_faces = {}
    
    def get_all_faces(self):
        """Get list of all registered faces"""
        return list(self.known_faces.keys())
    
    def delete_face(self, name):
        """Delete a face from database"""
        if name in self.known_faces:
            del self.known_faces[name]
            self.save_database()
            logger.info(f"Deleted: {name}")
            return True
        return False
