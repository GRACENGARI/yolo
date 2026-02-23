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
    def __init__(self, model_name='buffalo_s', min_confidence=0.5, device='cuda'):
        # Model Quantization: buffalo_s is the smaller/faster model for edge deployment
        self.known_faces = {} # { "name": embedding_vector }
        self.enabled = HAS_INSIGHTFACE
        
        if not self.enabled:
            logger.warning("InsightFace not installed. Face ID will run in MOCK mode.")
            return

        try:
            # Model Quantization & Edge Optimization: Handle explicit device toggle
            if device == 'cuda' or device == 'gpu':
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
                
            self.app = FaceAnalysis(name=model_name, providers=providers)
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info(f"ArcFace Model '{model_name}' (Quantized/Small) initialized on {providers[0]}.")
        except Exception as e:
            logger.error(f"Failed to load InsightFace models: {e}. Falling back to CPU.")
            try:
                self.app = FaceAnalysis(name=model_name, providers=['CPUExecutionProvider'])
                self.app.prepare(ctx_id=0, det_size=(640, 640))
            except:
                logger.error("Fail-Soft: InsightFace totally failed. Disabling biometric ID.")
                self.enabled = False

        self._load_mock_db()

    def _load_mock_db(self):
        pass

    def register_face(self, image, name):
        """
        Registers a face from an image (numpy array) with a name.
        """
        if not self.enabled:
            self.known_faces[name] = np.random.rand(512)
            return True

        try:
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
        except Exception as e:
            logger.error(f"Error during face registration: {e}")
            return False

    def identify(self, face_crop, threshold=0.4):
        """
        Input: Cropped face image (H, W, 3)
        Output: (Name, Confidence, Embedding) or (None, 0, None)
        """
        if not self.enabled:
            emb = np.random.rand(512)
            if np.random.random() > 0.98:
                return "Juma Macharia", 0.99, emb
            return "Unknown", 0.0, emb

        try:
            faces = self.app.get(face_crop)
            if len(faces) == 0:
                return "Unknown", 0.0, None
            
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
                return best_name, float(best_score), target_embedding
            
            return "Unknown", float(best_score), target_embedding
        except Exception as e:
            logger.error(f"Error during face identification: {e}")
        
        return "Unknown", 0.0, None

    def get_attributes(self, face_crop):
        """
        Returns age/gender if available
        """
        if not self.enabled:
            return {"gender": "Male", "age": 25}
            
        try:
            faces = self.app.get(face_crop)
            if not faces:
                return {}
            
            f = faces[0]
            return {
                "gender": "Male" if f.gender == 1 else "Female",
                "age": int(f.age)
            }
        except:
            return {}

