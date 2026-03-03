"""
MWEWE YOLOv8 Training Script
Fine-tune YOLOv8 on local vehicle dataset for improved classification
"""
from ultralytics import YOLO
import argparse
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TRAIN_YOLO")

def validate_dataset(data_path):
    """Validate dataset structure before training"""
    data_path = Path(data_path)
    
    if not data_path.exists():
        logger.error(f"Dataset config not found: {data_path}")
        return False
    
    # Check for images and labels
    base_dir = data_path.parent
    train_images = base_dir / 'images' / 'train'
    train_labels = base_dir / 'labels' / 'train'
    
    if not train_images.exists():
        logger.error(f"Training images directory not found: {train_images}")
        return False
    
    if not train_labels.exists():
        logger.error(f"Training labels directory not found: {train_labels}")
        return False
    
    # Count files
    image_count = len(list(train_images.glob('*.jpg'))) + len(list(train_images.glob('*.png')))
    label_count = len(list(train_labels.glob('*.txt')))
    
    logger.info(f"Found {image_count} training images")
    logger.info(f"Found {label_count} training labels")
    
    if image_count == 0:
        logger.error("No training images found!")
        logger.info("Add images to: training_data/vehicles/images/train/")
        return False
    
    if label_count == 0:
        logger.error("No training labels found!")
        logger.info("Add labels to: training_data/vehicles/labels/train/")
        logger.info("Use annotation tools like CVAT or LabelImg")
        return False
    
    if image_count != label_count:
        logger.warning(f"Mismatch: {image_count} images but {label_count} labels")
    
    return True

def train_mwewe_model(epochs=50, batch_size=16, device='cpu', resume=False, pretrained='yolov8n.pt'):
    """
    Train MWEWE vehicle detection model
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size (reduce if OOM)
        device: 'cpu' or 'cuda'
        resume: Resume from last checkpoint
        pretrained: Base model to start from
    """
    # Path to custom dataset YAML
    data_path = "training_data/vehicles/mwewe_vehicles.yaml"
    
    # Validate dataset
    if not validate_dataset(data_path):
        logger.error("Dataset validation failed. Fix errors and try again.")
        return
    
    # Load the base YOLOv8 model
    if resume:
        logger.info("Resuming training from last checkpoint...")
        model = YOLO("runs/detect/mwewe_vehicles_v1/weights/last.pt")
    else:
        logger.info(f"Loading base model: {pretrained}")
        model = YOLO(pretrained)
    
    logger.info(f"Starting MWEWE fine-tuning on {device}...")
    logger.info(f"Training for {epochs} epochs with batch size {batch_size}")
    
    # Train the model
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        device=device,
        name="mwewe_vehicles_v1",
        exist_ok=True,  # Allow overwriting
        
        # Optimizer settings
        optimizer='Adam',
        lr0=0.001,  # Initial learning rate
        lrf=0.01,  # Final learning rate (lr0 * lrf)
        momentum=0.937,
        weight_decay=0.0005,
        
        # Data Augmentation (Crucial for different angles and conditions)
        hsv_h=0.015,  # Hue variation
        hsv_s=0.7,    # Saturation variation
        hsv_v=0.4,    # Value/brightness variation
        degrees=10.0,  # Rotation (+/- degrees)
        translate=0.1, # Translation (+/- fraction)
        scale=0.5,     # Scaling (+/- gain)
        shear=0.0,     # Shear (+/- degrees)
        perspective=0.001,  # Perspective distortion
        flipud=0.0,    # Flip up-down probability
        fliplr=0.5,    # Flip left-right probability
        mosaic=1.0,    # Mosaic augmentation probability
        mixup=0.1,     # Mixup augmentation probability
        
        # Training settings
        patience=10,   # Early stopping patience
        save=True,     # Save checkpoints
        save_period=5, # Save every N epochs
        
        # Validation
        val=True,
        plots=True,    # Generate training plots
    )
    
    logger.info("✓ Training Complete!")
    logger.info(f"Best model: runs/detect/mwewe_vehicles_v1/weights/best.pt")
    logger.info(f"Last model: runs/detect/mwewe_vehicles_v1/weights/last.pt")
    
    # Validation metrics
    logger.info("\nValidation Metrics:")
    logger.info(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    logger.info(f"mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    
    return results

def export_model(model_path='runs/detect/mwewe_vehicles_v1/weights/best.pt', format='onnx'):
    """
    Export trained model to different formats
    
    Args:
        model_path: Path to trained model
        format: Export format (onnx, torchscript, tflite, etc.)
    """
    logger.info(f"Exporting model to {format}...")
    
    model = YOLO(model_path)
    model.export(format=format)
    
    logger.info(f"✓ Model exported successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MWEWE YOLOv8 Training Script")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (reduce if OOM)")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--pretrained", type=str, default="yolov8n.pt", help="Base model to start from")
    parser.add_argument("--export", action="store_true", help="Export trained model")
    parser.add_argument("--export-format", type=str, default="onnx", help="Export format")
    parser.add_argument("--validate-only", action="store_true", help="Only validate dataset structure")
    
    args = parser.parse_args()
    
    if args.validate_only:
        logger.info("Validating dataset structure...")
        validate_dataset("training_data/vehicles/mwewe_vehicles.yaml")
    elif args.export:
        export_model(format=args.export_format)
    else:
        train_mwewe_model(
            epochs=args.epochs, 
            batch_size=args.batch, 
            device=args.device,
            resume=args.resume,
            pretrained=args.pretrained
        )
