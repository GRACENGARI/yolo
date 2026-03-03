"""
Training Data Organization Tool
Helps structure datasets for YOLO and ReID training
"""
import os
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DATA_ORGANIZER")

class DatasetOrganizer:
    """
    Organizes training data into proper structure for YOLO and ReID training
    """
    
    def __init__(self, base_dir='training_data'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def create_yolo_structure(self):
        """
        Create YOLO dataset structure:
        training_data/
        ├── vehicles/
        │   ├── images/
        │   │   ├── train/
        │   │   └── val/
        │   ├── labels/
        │   │   ├── train/
        │   │   └── val/
        │   └── mwewe_vehicles.yaml
        """
        logger.info("Creating YOLO dataset structure...")
        
        vehicle_dir = self.base_dir / 'vehicles'
        
        # Create directories
        dirs = [
            vehicle_dir / 'images' / 'train',
            vehicle_dir / 'images' / 'val',
            vehicle_dir / 'labels' / 'train',
            vehicle_dir / 'labels' / 'val'
        ]
        
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created: {d}")
        
        # Create YAML config if it doesn't exist
        yaml_path = vehicle_dir / 'mwewe_vehicles.yaml'
        if not yaml_path.exists():
            yaml_content = """# MWEWE Vehicle Dataset Configuration
path: ../training_data/vehicles  # Dataset root dir
train: images/train  # Train images (relative to 'path')
val: images/val  # Val images (relative to 'path')

# Classes (local vehicle types)
names:
  0: Probox
  1: Matatu
  2: Boda-Boda
  3: Tuk-Tuk
  4: Saloon
  5: SUV
  6: Truck
  7: Bus

# Training parameters
nc: 8  # Number of classes
"""
            yaml_path.write_text(yaml_content)
            logger.info(f"Created: {yaml_path}")
    
    def create_reid_structure(self):
        """
        Create ReID dataset structure:
        training_data/
        ├── reid_persons/
        │   ├── train/
        │   │   ├── person_001/
        │   │   ├── person_002/
        │   │   └── ...
        │   └── val/
        │       ├── person_001/
        │       └── ...
        └── reid_vehicles/
            ├── train/
            └── val/
        """
        logger.info("Creating ReID dataset structure...")
        
        # Person ReID
        person_dir = self.base_dir / 'reid_persons'
        (person_dir / 'train').mkdir(parents=True, exist_ok=True)
        (person_dir / 'val').mkdir(parents=True, exist_ok=True)
        logger.info(f"Created: {person_dir}")
        
        # Vehicle ReID
        vehicle_dir = self.base_dir / 'reid_vehicles'
        (vehicle_dir / 'train').mkdir(parents=True, exist_ok=True)
        (vehicle_dir / 'val').mkdir(parents=True, exist_ok=True)
        logger.info(f"Created: {vehicle_dir}")
        
        # Create README
        readme_path = self.base_dir / 'REID_README.md'
        readme_content = """# ReID Dataset Structure

## Person ReID
Place person images in subdirectories named by person ID:
```
reid_persons/
├── train/
│   ├── person_001/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   ├── person_002/
│   └── ...
└── val/
    └── ...
```

## Vehicle ReID
Same structure for vehicles:
```
reid_vehicles/
├── train/
│   ├── vehicle_001/
│   │   ├── front.jpg
│   │   ├── side.jpg
│   │   └── rear.jpg
│   └── ...
└── val/
```

Each subdirectory represents one identity (person or vehicle).
Multiple images per identity help the model learn invariant features.
"""
        readme_path.write_text(readme_content)
        logger.info(f"Created: {readme_path}")
    
    def create_faces_structure(self):
        """
        Create face recognition dataset structure:
        training_data/
        └── faces/
            ├── known_subjects/
            │   ├── juma_macharia/
            │   ├── suspect_002/
            │   └── ...
            └── unknown/
        """
        logger.info("Creating face dataset structure...")
        
        faces_dir = self.base_dir / 'faces'
        (faces_dir / 'known_subjects').mkdir(parents=True, exist_ok=True)
        (faces_dir / 'unknown').mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created: {faces_dir}")
        
        # Create README
        readme_path = faces_dir / 'README.md'
        readme_content = """# Face Recognition Dataset

## Structure
```
faces/
├── known_subjects/
│   ├── juma_macharia/
│   │   ├── face_001.jpg
│   │   ├── face_002.jpg
│   │   └── ...
│   └── suspect_002/
└── unknown/
    └── unidentified_faces.jpg
```

## Guidelines
- Each known subject gets their own folder
- Include multiple angles and lighting conditions
- Minimum 5 images per subject recommended
- Images should be clear face crops (not full body)
- Use consistent naming: subject_name/face_NNN.jpg
"""
        readme_path.write_text(readme_content)
        logger.info(f"Created: {readme_path}")
    
    def validate_yolo_dataset(self):
        """Validate YOLO dataset structure"""
        logger.info("Validating YOLO dataset...")
        
        vehicle_dir = self.base_dir / 'vehicles'
        
        # Check for images and labels
        train_images = list((vehicle_dir / 'images' / 'train').glob('*.jpg'))
        train_labels = list((vehicle_dir / 'labels' / 'train').glob('*.txt'))
        
        logger.info(f"Found {len(train_images)} training images")
        logger.info(f"Found {len(train_labels)} training labels")
        
        if len(train_images) == 0:
            logger.warning("No training images found! Add images to training_data/vehicles/images/train/")
        
        if len(train_labels) == 0:
            logger.warning("No training labels found! Add labels to training_data/vehicles/labels/train/")
        
        # Check for mismatches
        image_stems = {img.stem for img in train_images}
        label_stems = {lbl.stem for lbl in train_labels}
        
        missing_labels = image_stems - label_stems
        if missing_labels:
            logger.warning(f"{len(missing_labels)} images missing labels")
        
        missing_images = label_stems - image_stems
        if missing_images:
            logger.warning(f"{len(missing_images)} labels missing images")
        
        return len(train_images) > 0 and len(train_labels) > 0
    
    def setup_all(self):
        """Create all dataset structures"""
        logger.info("Setting up complete training data structure...")
        self.create_yolo_structure()
        self.create_reid_structure()
        self.create_faces_structure()
        logger.info("✓ Training data structure created successfully")
        logger.info(f"Base directory: {self.base_dir.absolute()}")


if __name__ == '__main__':
    organizer = DatasetOrganizer()
    organizer.setup_all()
    organizer.validate_yolo_dataset()
