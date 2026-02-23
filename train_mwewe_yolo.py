from ultralytics import YOLO
import argparse
import os

def train_mwewe_model(epochs=50, batch_size=16, device='cpu'):
    # Load the base YOLOv8 model (Nano is best for edge performance)
    model = YOLO("yolov8n.pt")
    
    # Path to your custom dataset YAML
    data_path = "training_data/vehicles/mwewe_vehicles.yaml"
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset config not found at {data_path}")
        return

    print(f"Starting MWEWE fine-tuning on {device}...")
    
    # Train the model
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        device=device,
        name="mwewe_vehicles_v1",
        # Data Augmentation (Crucial for different angles and conditions)
        hsv_h=0.015, # Hue
        hsv_s=0.7, # Saturation
        hsv_v=0.4, # Value
        degrees=10.0, # Rotate
        translate=0.1, # Move
        scale=0.5, # Scale
        shear=0.0, # Shear
        perspective=0.001, # Perspective
        flipud=0.0, # Flip up-down
        fliplr=0.5, # Flip left-right
        mosaic=1.0, # Mosaic (multiple images in one)
    )
    
    print("Training Complete. Model saved in runs/detect/mwewe_vehicles_v1/weights/best.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MWEWE YOLOv8 Training Script")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    
    args = parser.parse_args()
    
    train_mwewe_model(
        epochs=args.epochs, 
        batch_size=args.batch, 
        device=args.device
    )
