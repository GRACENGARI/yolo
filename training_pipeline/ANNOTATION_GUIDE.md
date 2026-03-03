# HAWKEYE Annotation Guide

## Overview
This guide explains how to annotate images for YOLO training using free annotation tools.

## Recommended Tools

### 1. LabelImg (Easiest for Beginners)
**Best for:** Quick bounding box annotation

**Installation:**
```bash
pip install labelImg
```

**Usage:**
```bash
labelImg training_data/vehicles/images/train training_data/vehicles/classes.txt
```

**Workflow:**
1. Open LabelImg
2. Click "Open Dir" → Select `training_data/vehicles/images/train`
3. Click "Change Save Dir" → Select `training_data/vehicles/labels/train`
4. Set format to "YOLO" (not PascalVOC)
5. Draw boxes around vehicles
6. Assign class labels
7. Press 'Ctrl+S' to save
8. Press 'D' to move to next image

**Classes:**
- 0: Probox
- 1: Matatu
- 2: Boda-Boda
- 3: Tuk-Tuk
- 4: Saloon
- 5: SUV
- 6: Truck
- 7: Bus

### 2. CVAT (Best for Team Collaboration)
**Best for:** Large datasets, multiple annotators, cloud storage

**Installation (Docker):**
```bash
docker run -d -p 8080:8080 --name cvat openvino/cvat
```

**Access:** http://localhost:8080

**Workflow:**
1. Create new task
2. Upload images
3. Add labels (Probox, Matatu, etc.)
4. Annotate using rectangle tool
5. Export in "YOLO 1.1" format
6. Extract to `training_data/vehicles/`

### 3. Roboflow (Cloud-Based)
**Best for:** Auto-annotation, augmentation, version control

**Website:** https://roboflow.com

**Workflow:**
1. Create free account
2. Create new project (Object Detection)
3. Upload images
4. Use Smart Polygon or manual annotation
5. Apply augmentations (rotation, brightness, etc.)
6. Export in "YOLOv8" format
7. Download and extract to `training_data/vehicles/`

## Annotation Best Practices

### Quality Guidelines
1. **Tight Boxes:** Draw boxes as close to the vehicle as possible
2. **Complete Objects:** Include the entire vehicle, don't cut off parts
3. **Occlusion:** If vehicle is >50% visible, annotate it
4. **Multiple Angles:** Annotate front, side, rear, and angled views
5. **Consistency:** Use the same class for the same vehicle type

### Dataset Balance
Aim for balanced representation:
- Each class: 100-500 images minimum
- Multiple lighting conditions (day, night, dusk)
- Various weather (sunny, rainy, foggy)
- Different angles (0°, 45°, 90°, 135°, 180°)
- Multiple distances (close-up, medium, far)

### Label Format (YOLO)
Each `.txt` file should have one line per object:
```
<class_id> <x_center> <y_center> <width> <height>
```

All values are normalized (0-1):
- `class_id`: Integer (0-7 for our classes)
- `x_center`: Center X / Image Width
- `y_center`: Center Y / Image Height
- `width`: Box Width / Image Width
- `height`: Box Height / Image Height

**Example:**
```
1 0.5 0.5 0.3 0.4
```
This is a Matatu (class 1) centered in the image, taking up 30% width and 40% height.

## Validation

After annotation, validate your dataset:
```bash
python train_mwewe_yolo.py --validate-only
```

This checks:
- Image/label count match
- Label format correctness
- Class distribution
- Missing files

## Tips for Efficiency

1. **Keyboard Shortcuts:** Learn tool shortcuts to speed up annotation
2. **Batch Processing:** Annotate similar images together
3. **Auto-Annotation:** Use pre-trained YOLO to generate initial boxes, then correct
4. **Quality Over Quantity:** 200 perfect annotations > 1000 sloppy ones
5. **Regular Breaks:** Annotation fatigue leads to errors

## Auto-Annotation Script

Use existing YOLO model to pre-annotate:
```bash
python training_pipeline/auto_annotate.py --source training_data/vehicles/images/train
```

Then manually review and correct the generated labels.

## Troubleshooting

**Problem:** Labels not saving
- **Solution:** Check write permissions on labels directory

**Problem:** Wrong format exported
- **Solution:** Ensure "YOLO" format is selected, not PascalVOC or COCO

**Problem:** Classes not matching
- **Solution:** Verify `classes.txt` matches `mwewe_vehicles.yaml`

**Problem:** Training fails with "no labels found"
- **Solution:** Check that `.txt` files are in `labels/train/`, not `images/train/`

## Resources

- LabelImg: https://github.com/heartexlabs/labelImg
- CVAT: https://github.com/opencv/cvat
- Roboflow: https://roboflow.com
- YOLO Format: https://docs.ultralytics.com/datasets/detect/
