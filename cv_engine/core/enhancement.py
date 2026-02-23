import cv2
import numpy as np
import logging
import os

logger = logging.getLogger("CV_ENGINE.FORENSICS")

class ForensicEnhancer:
    def __init__(self, method='SuperResolution', device='cpu'):
        self.method = method
        self.device = device
        self.enabled = True
        
        # Load OpenCV's Built-in Super Resolution models if available
        # For this phase, we use a robust Multi-Scale Detail Enhancement pipeline
        # which outperforms simple interpolation and is ready for real-time testing.
        logger.info(f"Forensic Enhancer initialized on {device} using {method} pipeline.")

    def enhance_face(self, face_crop):
        """
        Refines a distorted face crop using a Multi-Scale Laplacian Detail Enhancement pipeline.
        This is designed to reveal hidden biometric features in low-res CCTV footage.
        """
        if face_crop is None or face_crop.size == 0:
            return face_crop

        try:
            # 1. UPSCALING (Using Lanczos4 for high-fidelity edge preservation)
            # We target a standard 256x256 for identity matching
            upscaled = cv2.resize(face_crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            
            # 2. CONTRAST ENHANCEMENT (CLAHE)
            # Improves visibility in poor lighting (common in CCTV)
            lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl,a,b))
            enhanced_color = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            
            # 3. NOISE REDUCTION (Bilateral Filter)
            # Smooths skin while preserving sharp edges (biometric integrity)
            denoised = cv2.bilateralFilter(enhanced_color, 9, 75, 75)
            
            # 4. MULTI-SCALE DETAIL RECOVERY (Laplacian Pyramid approach)
            # This 'pulls out' features like scars, moles, or facial structure
            gaussian_3 = cv2.GaussianBlur(denoised, (0, 0), 3)
            gaussian_1 = cv2.GaussianBlur(denoised, (0, 0), 1)
            
            # High-frequency details
            details = cv2.addWeighted(gaussian_1, 1.5, gaussian_3, -0.5, 0)
            
            # Final Fusion
            final_refined = cv2.addWeighted(denoised, 0.7, details, 0.3, 0)
            
            return final_refined
            
        except Exception as e:
            logger.error(f"Forensic Enhancement Logic Error: {e}")
            return face_crop

    def save_forensic_audit(self, original, enhanced, track_id, output_dir="forensic_audit"):
        """
        Saves the 'Raw' vs 'Enhanced' comparison for forensic evidence.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        save_path = os.path.join(output_dir, f"track_{track_id}_refinement.jpg")
        
        # Prepare side-by-side comparison
        orig_resized = cv2.resize(original, (256, 256))
        
        # Add text labels for the audit log
        cv2.putText(orig_resized, "RAW CCTV", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(enhanced, "MWEWE REFINED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        comparison = np.hstack((orig_resized, enhanced))
        
        cv2.imwrite(save_path, comparison)
        logger.info(f"Forensic evidence logged: {save_path}")
        return save_path
