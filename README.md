# 🦅 HAWKEYE - AI Vehicle Surveillance System

Real-time vehicle detection, tracking, and forensic analysis using YOLOv8.

## 📊 Live Data Results

| Metric | Value |
|--------|-------|
| Total Vehicle Sightings | 64,001 |
| Unique Vehicles Tracked | 883 |
| Detection Confidence | 50–95% |
| Tracking Period | Feb 28 – Mar 12, 2026 |
| Processing Speed | 4–6 FPS (CPU) |
| Detection Model | YOLOv8n |

## 🎯 What It Does

- **Real-time detection** of cars, motorcycles, buses, trucks
- **Target registration** — mark any vehicle as a suspect, it turns RED with a movement trail
- **Forensic search** — find every sighting of any vehicle with timestamps
- **Movement trail** — reconstruct the full path a vehicle took
- **Analytics dashboard** — live stats, top active vehicles, recent sightings
- **Data export** — export all 64,001 records to CSV for evidence

## 🚀 Run the System

```bash
pip install ultralytics flask opencv-python
python run_vehicle_simple.py
```

Open browser: **http://localhost:5001**

## 📁 Data Evidence

The file `vehicle_evidence_report.csv` contains all 64,001 vehicle sightings with:
- Vehicle ID
- Timestamp (precise to the second)
- Camera ID
- Location
- Detection confidence score
- Bounding box coordinates

## 🗄️ Database

```bash
# View statistics
python view_database.py stats

# Recent sightings
python view_database.py recent 20

# Search specific vehicle
python view_database.py search "Vehicle-55"

# Export to CSV
python view_database.py export report.csv
```

## 🏗️ Architecture

```
cv-engine/
├── run_vehicle_simple.py     # Main tracker + web interface
├── mini_backend/
│   ├── sighting_tracker.py   # SQLite database handler
│   └── sightings.db          # 64,001 sightings stored here
├── predictive_analytics.py   # Pattern analysis engine
├── view_database.py          # CLI database viewer
└── vehicle_evidence_report.csv  # Exported evidence data
```

## 🔍 Top 10 Most Tracked Vehicles

| Rank | Vehicle | Sightings |
|------|---------|-----------|
| 1 | Vehicle-55 | 1,089 |
| 2 | Vehicle-46 | 1,088 |
| 3 | Vehicle-104 | 928 |
| 4 | Vehicle-162 | 893 |
| 5 | Vehicle-113 | 890 |
| 6 | Vehicle-10 | 884 |
| 7 | Vehicle-171 | 871 |
| 8 | Vehicle-2 | 842 |
| 9 | Vehicle-220 | 768 |
| 10 | Vehicle-229 | 683 |

## 🛠️ Tech Stack

- **YOLOv8** — object detection and tracking
- **OpenCV** — video processing
- **Flask** — web interface
- **SQLite** — persistent data storage
- **Python** — core engine
