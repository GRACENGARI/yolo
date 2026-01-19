import numpy as np
import cv2
import logging
import os

# Try to import InsightFace, handle failure gracefully
try:
    import insightface
    from insightface.app import FaceAnalysis
    HAS_INSIGHTFACE = True
except ImportError:
    HAS_INSIGHTFACE = False

logger = logging.getLogger("CV_ENGINE.FACE")

class FaceIdentifier:
    def __init__(self, model_name='buffalo_l', min_confidence=0.5):
        self.known_faces = {} # { "name": embedding_vector }
        self.enabled = HAS_INSIGHTFACE
        
        if not self.enabled:
            logger.warning("InsightFace not installed. Face ID will run in MOCK mode.")
            return

        try:
            # Initialize FaceAnalysis (Detection + Recognition)
            # providers=['CPUExecutionProvider'] forces CPU mode for compatibility
            self.app = FaceAnalysis(name=model_name, providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info(f"ArcFace Model '{model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load InsightFace models: {e}")
            self.enabled = False

        # Load some dummy known faces or allow registration
        # In a real app, load from disk/DB
        self._load_mock_db()

    def _load_mock_db(self):
        # We can pre-load embeddings here if we had them.
        # For now, we will learn as we go or expect external registration.
        pass

    def register_face(self, image, name):
        """
        Registers a face from an image (numpy array) with a name.
        """
        if not self.enabled:
            self.known_faces[name] = np.random.rand(512)
            return True

        faces = self.app.get(image)
        if len(faces) == 0:
            logger.warning(f"No face detected for registration: {name}")
            return False
        
        # Take the largest face
        faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
        embedding = faces[0].embedding
        self.known_faces[name] = embedding
        logger.info(f"Registered new identity: {name}")
        return True

    def identify(self, face_crop, threshold=0.4):
        """
        Input: Cropped face image (H, W, 3)
        Output: (Name, Confidence) or (None, 0)
        """
        if not self.enabled:
            # Mock Logic: Randomly identify 'Juma' sometimes
            if np.random.random() > 0.95:
                return "Juma Macharia", 0.99
            return "Unknown", 0.0

        # InsightFace expects the full image usually, but if we pass a crop, 
        # the detector might fail if the crop is too tight or lacks context.
        # However, for efficiency in a tracking pipeline, we usually pass the crop 
        # directly to the recognition model if we already have the bbox.
        # BUT, `FaceAnalysis` pipeline runs Det -> Rec.
        # If we already have the bbox from YOLO, we can just run the recognition model?
        # The `app.get` runs both. 
        # Optimization: We'll pass the crop to `app.get`.
        
        faces = self.app.get(face_crop)
        if len(faces) == 0:
            return "Unknown", 0.0
        
        # Take the most prominent face in the crop
        target_embedding = faces[0].embedding
        
        best_score = -1.0
        best_name = "Unknown"

        for name, known_embedding in self.known_faces.items():
            # Cosine Similarity
            score = np.dot(target_embedding, known_embedding) / (np.linalg.norm(target_embedding) * np.linalg.norm(known_embedding))
            if score > best_score:
                best_score = score
                best_name = name

        if best_score > threshold:
            return best_name, float(best_score)
        
        return "Unknown", float(best_score)

    def get_attributes(self, face_crop):
        """
        Returns age/gender if available
        """
        if not self.enabled:
            return {"gender": "Male", "age": 25}
            
        faces = self.app.get(face_crop)
        if not faces:
            return {}
        
        f = faces[0]
        return {
            "gender": "Male" if f.gender == 1 else "Female",
            "age": int(f.age)
        }
