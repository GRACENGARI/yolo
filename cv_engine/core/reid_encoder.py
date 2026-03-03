"""
Advanced ReID (Re-Identification) Encoder using OSNet
Replaces SimpleEncoder for persistent tracking across occlusions
"""
import numpy as np
import cv2
import logging
import os

logger = logging.getLogger("CV_ENGINE.REID")

class ReIDEncoder:
    """
    Advanced Re-Identification encoder using OSNet (Omni-Scale Network)
    Provides robust feature extraction for person/vehicle tracking
    """
    
    def __init__(self, model_name='osnet_x1_0', device='cpu'):
        self.model_name = model_name
        self.device = device
        self.enabled = False
        self.model = None
        
        # Try to load torchreid (if available)
        try:
            import torch
            import torchreid
            
            self.torch = torch
            self.torchreid = torchreid
            
            logger.info(f"Loading ReID model: {model_name} on {device}")
            
            # Load pretrained OSNet model
            self.model = torchreid.models.build_model(
                name=model_name,
                num_classes=1000,  # Pretrained on large dataset
                loss='softmax',
                pretrained=True
            )
            
            # Set device
            if device in ['cuda', 'gpu'] and torch.cuda.is_available():
                self.model = self.model.cuda()
                self.model.half()  # FP16 for GPU
                logger.info(f"ReID model loaded on CUDA (FP16)")
            else:
                self.model = self.model.cpu()
                logger.info(f"ReID model loaded on CPU")
            
            self.model.eval()
            self.enabled = True
            
            # Preprocessing parameters (ImageNet normalization)
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
            self.input_size = (256, 128)  # Height x Width for person ReID
            
        except ImportError:
            logger.warning("torchreid not installed. Falling back to SimpleEncoder.")
            logger.info("Install with: pip install torchreid")
        except Exception as e:
            logger.error(f"Failed to load ReID model: {e}")
            logger.warning("Falling back to SimpleEncoder")
    
    def preprocess(self, img):
        """
        Preprocess image for ReID model
        
        Args:
            img: BGR image (H, W, 3)
        
        Returns:
            Preprocessed tensor
        """
        # Resize to model input size
        img = cv2.resize(img, (self.input_size[1], self.input_size[0]))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        img = (img - self.mean) / self.std
        
        # Convert to tensor (C, H, W)
        img = img.transpose(2, 0, 1)
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return self.torch.from_numpy(img).float()
    
    def extract_features(self, img):
        """
        Extract ReID features from a single image
        
        Args:
            img: BGR image (H, W, 3)
        
        Returns:
            Feature vector (numpy array)
        """
        if not self.enabled:
            # Fallback to simple color histogram
            return self._simple_feature(img)
        
        try:
            # Preprocess
            tensor = self.preprocess(img)
            
            # Move to device
            if self.device in ['cuda', 'gpu'] and self.torch.cuda.is_available():
                tensor = tensor.cuda()
                if hasattr(tensor, 'half'):
                    tensor = tensor.half()
            
            # Extract features
            with self.torch.no_grad():
                features = self.model(tensor)
            
            # Convert to numpy
            features = features.cpu().numpy().flatten()
            
            # L2 normalize
            features = features / (np.linalg.norm(features) + 1e-12)
            
            return features
            
        except Exception as e:
            logger.error(f"ReID feature extraction failed: {e}")
            return self._simple_feature(img)
    
    def _simple_feature(self, img):
        """
        Fallback simple feature extraction (color histogram)
        Used when ReID model is unavailable
        """
        if img.size == 0:
            return np.random.rand(128)
        
        # Resize to standard size
        img = cv2.resize(img, (64, 128))
        
        # Compute color histogram
        hist_b = cv2.calcHist([img], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([img], [1], None, [32], [0, 256])
        hist_r = cv2.calcHist([img], [2], None, [32], [0, 256])
        
        # Concatenate and normalize
        feature = np.concatenate([hist_b, hist_g, hist_r]).flatten()
        feature = feature / (np.sum(feature) + 1e-12)
        
        return feature
    
    def __call__(self, frame, boxes):
        """
        Extract features for multiple bounding boxes
        
        Args:
            frame: Full frame image (H, W, 3)
            boxes: List of bounding boxes in [x, y, w, h] format
        
        Returns:
            numpy array of features (N, feature_dim)
        """
        features = []
        
        for box in boxes:
            x, y, w, h = map(int, box)
            
            # Clamp to frame boundaries
            x = max(0, x)
            y = max(0, y)
            x2 = min(frame.shape[1], x + w)
            y2 = min(frame.shape[0], y + h)
            
            # Extract ROI
            roi = frame[y:y2, x:x2]
            
            if roi.size == 0:
                # Empty ROI, use random feature
                features.append(np.random.rand(512 if self.enabled else 128))
                continue
            
            # Extract features
            feature = self.extract_features(roi)
            features.append(feature)
        
        return np.array(features)
    
    def get_feature_dim(self):
        """Get the dimensionality of extracted features"""
        if self.enabled:
            return 512  # OSNet feature dimension
        else:
            return 128  # Simple feature dimension


class SimpleEncoder:
    """
    Fallback simple encoder (original implementation)
    Used when ReID model is not available
    """
    
    def __call__(self, frame, boxes):
        features = []
        for box in boxes:
            x, y, w, h = map(int, box)
            x, y = max(0, x), max(0, y)
            roi = frame[y:y+h, x:x+w]
            
            if roi.size == 0:
                features.append(np.random.rand(128))
                continue
            
            # Simple color-based feature
            avg_color = cv2.resize(roi, (8, 8)).flatten()
            feature = avg_color / 255.0
            features.append(feature)
        
        return np.array(features)
