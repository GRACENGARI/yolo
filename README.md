# HAWKEYE Vision Engine (The Eyes)

**Part of Project Mwewe // National Security Surveillance Grid**

The **Vision Engine** is the edge-processing node of the HAWKEYE system. It utilizes Computer Vision (CV) to analyze video streams in real-time, detecting vehicles and persons of interest, and transmitting "sightings" to the central grid.

## 👁️ Capabilities

1.  **Object Detection**: Uses **YOLOv8** (You Only Look Once) for high-speed detection of people and vehicles.
2.  **Target Tracking**: Implements **DeepSORT** to assign unique IDs to subjects and track their trajectory across frames.
3.  **Feature Extraction**: Analyzes color and basic attributes (e.g., "Red Truck").
4.  **Grid Integration**: Pushes structured JSON telemetry to the Backend API.

## 🛠️ Technology Stack

*   **Core**: Python 3.12+
*   **Model**: Ultralytics YOLOv8 (`yolov8n.pt`)
*   **Tracking**: DeepSORT
*   **Math**: NumPy, OpenCV

## 🚀 Mission Start

### Prerequisites
*   Python 3.10 or higher.
*   A video source (default is `people.mp4` for simulation).

### Installation

```bash
cd cv-engine
pip install -r cv_engine/requirements.txt
```

### Running the Engine

You can run the engine directly or use the provided batch script in the project root.

**Direct Execution:**
```bash
python cv_engine/stream_processor.py
```

**Configuration:**
*   The script defaults to processing `people.mp4`.
*   To use a webcam, modify the source in `stream_processor.py` to `0` or `1`.

## 📂 Key Files

*   `stream_processor.py`: Main entry point. Captures frames, runs inference, and sends data.
*   `yolov8n.pt`: Pre-trained detection model weights.
*   `cv_out.log`: Local logs for debugging detection events.

---
*Classified: INTERNAL USE ONLY*
