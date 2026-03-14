"""
HAWKEYE - Process people.mp4 and export all person detections to CSV
"""
import cv2
import csv
from datetime import datetime, timedelta
from ultralytics import YOLO

VIDEO_FILE = "people.mp4"
OUTPUT_CSV = "people_surveillance_report.csv"

print("=" * 60)
print("  HAWKEYE - PEOPLE SURVEILLANCE DATA EXPORT")
print("=" * 60)

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(VIDEO_FILE)

fps = cap.get(cv2.CAP_PROP_FPS) or 25
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration_secs = total_frames / fps

print(f"\nVideo: {VIDEO_FILE}")
print(f"FPS: {fps:.1f} | Frames: {total_frames} | Duration: {duration_secs:.1f}s")
print(f"\nProcessing... (this may take a minute)")

# Use a fake base timestamp so data looks realistic
base_time = datetime(2026, 3, 12, 8, 0, 0)

rows = []
frame_num = 0
track_first_seen = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Only process every 3rd frame for speed
    if frame_num % 3 != 0:
        frame_num += 1
        continue

    results = model.track(frame, persist=True, verbose=False, classes=[0])  # class 0 = person

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()

        # Timestamp based on frame position
        frame_time = base_time + timedelta(seconds=frame_num / fps)

        for box, track_id, conf in zip(boxes, track_ids, confidences):
            if conf < 0.4:
                continue

            x1, y1, x2, y2 = map(int, box)
            w = x2 - x1
            h = y2 - y1

            if track_id not in track_first_seen:
                track_first_seen[track_id] = frame_time.strftime('%Y-%m-%d %H:%M:%S')

            rows.append({
                'Person_ID': f'Person-{track_id}',
                'Timestamp': frame_time.strftime('%Y-%m-%d %H:%M:%S'),
                'First_Seen': track_first_seen[track_id],
                'Camera_ID': 'CAM-PEOPLE-01',
                'Location': 'Surveillance Zone A',
                'Confidence_Pct': round(conf * 100, 1),
                'BBox_X': x1,
                'BBox_Y': y1,
                'BBox_W': w,
                'BBox_H': h,
                'Frame_Number': frame_num
            })

    frame_num += 1

    if frame_num % 100 == 0:
        pct = (frame_num / total_frames) * 100
        print(f"  Progress: {pct:.0f}% | Detections so far: {len(rows)}")

cap.release()

# Write CSV
with open(OUTPUT_CSV, 'w', newline='') as f:
    fieldnames = ['Person_ID', 'Timestamp', 'First_Seen', 'Camera_ID',
                  'Location', 'Confidence_Pct', 'BBox_X', 'BBox_Y', 'BBox_W', 'BBox_H', 'Frame_Number']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

# Summary
unique_people = len(set(r['Person_ID'] for r in rows))
print(f"\n{'=' * 60}")
print(f"  EXPORT COMPLETE")
print(f"{'=' * 60}")
print(f"  Total Detections : {len(rows)}")
print(f"  Unique People    : {unique_people}")
print(f"  Output File      : {OUTPUT_CSV}")
print(f"{'=' * 60}")
