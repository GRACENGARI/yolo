"""
Advanced Forensic Image Enhancement using GFPGAN/CodeFormer
Provides "Zoom & Enhance" capability for low-quality CCTV footage
"""
import cv2
import numpy as np
import logging
import os
from pathlib import Path

logger = logging.getLogger("CV_ENGINE.FORENSICS_V2")

class ForensicEnhancerV2:
    """
    Advanced forensic enhancement using state-of-the-art face restoration models
    Supports GFPGAN and CodeFormer for identity recovery from degraded footage
    """
    
    def __init__(self, model_type='gfpgan', device='cpu', save_audit=True, audit_dir='./forensic_audit'):
        self.model_type = model_type
        self.device = device
        self.save_audit = save_audit
        self.audit_dir = Path(audit_dir)
        self.enabled = False
        self.model = None
        
        # Create audit directory
        if self.save_audit:
            self.audit_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to load the requested model
        if model_type == 'gfpgan':
            self._load_gfpgan()
        elif model_type == 'codeformer':
            self._load_codeformer()
        else:
            logger.warning(f"Unknown model type: {model_type}. Using basic enhancement.")
    
    def _load_gfpgan(self):
        """Load GFPGAN model for face restoration"""
        try:
            from gfpgan import GFPGANer
            
            logger.info("Loading GFPGAN model...")
            
            # Model path (will auto-download if not present)
            model_path = 'weights/GFPGANv1.4.pth'
            
            self.model = GFPGANer(
                model_path=model_path,
                upscale=2,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None,  # No background upsampler for speed
                device=self.device
            )
            
            self.enabled = True
            logger.info("GFPGAN loaded successfully")
            
        except ImportError:
            logger.warning("GFPGAN not installed. Install with: pip install gfpgan")
            logger.info("Falling back to basic enhancement")
        except Exception as e:
            logger.error(f"Failed to load GFPGAN: {e}")
            logger.info("Falling back to basic enhancement")
    
    def _load_codeformer(self):
        """Load CodeFormer model for face restoration"""
        try:
            # CodeFormer integration (placeholder for future implementation)
            logger.warning("CodeFormer integration not yet implemented")
            logger.info("Falling back to basic enhancement")
        except Exception as e:
            logger.error(f"Failed to load CodeFormer: {e}")
    
    def enhance_face(self, face_crop, confidence=0.0):
        """
        Enhance a face crop using the loaded model
        
        Args:
            face_crop: BGR image of face (H, W, 3)
            confidence: Original detection confidence (for logging)
        
        Returns:
            Enhanced face image
        """
        if face_crop is None or face_crop.size == 0:
            return face_crop
        
        # Check minimum size
        if face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
            logger.debug("Face too small for enhancement")
            return face_crop
        
        try:
            if self.enabled and self.model_type == 'gfpgan':
                return self._enhance_with_gfpgan(face_crop)
            else:
                return self._enhance_basic(face_crop)
        except Exception as e:
            logger.error(f"Enhancement failed: {e}")
            return face_crop
    
    def _enhance_with_gfpgan(self, face_crop):
        """
        Enhance using GFPGAN model
        
        Args:
            face_crop: BGR image
        
        Returns:
            Enhanced BGR image
        """
        try:
            # GFPGAN expects RGB input
            input_img = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            
            # Run enhancement
            _, _, restored_face = self.model.enhance(
                input_img,
                has_aligned=False,
                only_center_face=True,
                paste_back=True
            )
            
            # Convert back to BGR
            if restored_face is not None:
                enhanced = cv2.cvtColor(restored_face, cv2.COLOR_RGB2BGR)
                logger.debug("GFPGAN enhancement successful")
                return enhanced
            else:
                logger.warning("GFPGAN returned None, using original")
                return face_crop
                
        except Exception as e:
            logger.error(f"GFPGAN enhancement error: {e}")
            return face_crop
    
    def _enhance_basic(self, face_crop):
        """
        Fallback basic enhancement using OpenCV
        Multi-scale detail enhancement pipeline
        """
        try:
            # 1. Upscale using Lanczos
            upscaled = cv2.resize(face_crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            
            # 2. CLAHE for contrast
            lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_color = cv2.merge((cl, a, b))
            enhanced_color = cv2.cvtColor(enhanced_color, cv2.COLOR_LAB2BGR)
            
            # 3. Bilateral filter for noise reduction
            denoised = cv2.bilateralFilter(enhanced_color, 9, 75, 75)
            
            # 4. Detail enhancement
            gaussian_3 = cv2.GaussianBlur(denoised, (0, 0), 3)
            gaussian_1 = cv2.GaussianBlur(denoised, (0, 0), 1)
            details = cv2.addWeighted(gaussian_1, 1.5, gaussian_3, -0.5, 0)
            
            # 5. Final fusion
            final = cv2.addWeighted(denoised, 0.7, details, 0.3, 0)
            
            return final
            
        except Exception as e:
            logger.error(f"Basic enhancement error: {e}")
            return face_crop
    
    def save_forensic_audit(self, original, enhanced, track_id, metadata=None):
        """
        Save forensic audit trail (raw vs enhanced comparison)
        Critical for legal chain of custody
        
        Args:
            original: Original face crop
            enhanced: Enhanced face crop
            track_id: Track ID for filename
            metadata: Optional dict with confidence, timestamp, etc.
        """
        if not self.save_audit:
            return None
        
        try:
            # Ensure both images are same size for comparison
            h, w = enhanced.shape[:2]
            orig_resized = cv2.resize(original, (w, h))
            
            # Add labels
            orig_labeled = orig_resized.copy()
            enh_labeled = enhanced.copy()
            
            cv2.putText(orig_labeled, "RAW CCTV", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(enh_labeled, "MWEWE ENHANCED", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add metadata if provided
            if metadata:
                y_offset = 60
                for key, value in metadata.items():
                    text = f"{key}: {value}"
                    cv2.putText(enh_labeled, text, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_offset += 25
            
            # Create side-by-side comparison
            comparison = np.hstack((orig_labeled, enh_labeled))
            
            # Save with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"track_{track_id}_{timestamp}_forensic.jpg"
            save_path = self.audit_dir / filename
            
            cv2.imwrite(str(save_path), comparison)
            logger.info(f"Forensic audit saved: {save_path}")
            
            return str(save_path)
            
        except Exception as e:
            logger.error(f"Failed to save forensic audit: {e}")
            return None
    
    def should_enhance(self, confidence, face_size):
        """
        Determine if enhancement should be triggered
        
        Args:
            confidence: Detection/recognition confidence (0-1)
            face_size: Size of face in pixels (min dimension)
        
        Returns:
            bool: True if enhancement should be applied
        """
        # Trigger logic from Phase 4 spec:
        # If Face_Confidence < 40% AND Face_Size > 20px
        if confidence < 0.4 and face_size > 20:
            return True
        
        # Also enhance very small faces regardless of confidence
        if face_size < 80:
            return True
        
        return False
    
    def batch_enhance(self, face_crops):
        """
        Enhance multiple faces in batch (for efficiency)
        
        Args:
            face_crops: List of face images
        
        Returns:
            List of enhanced face images
        """
        enhanced_faces = []
        
        for face in face_crops:
            enhanced = self.enhance_face(face)
            enhanced_faces.append(enhanced)
        
        return enhanced_faces
    
    def get_model_info(self):
        """Get information about the loaded model"""
        return {
            'model_type': self.model_type,
            'enabled': self.enabled,
            'device': self.device,
            'save_audit': self.save_audit,
            'audit_dir': str(self.audit_dir)
        }
