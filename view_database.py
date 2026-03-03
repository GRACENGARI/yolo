"""
HAWKEYE Database Viewer - View Historical Data
Shows all past vehicle sightings, statistics, and forensic data
"""
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

DB_PATH = "mini_backend/sightings.db"

def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def view_all_sightings(limit=50):
    """View recent sightings"""
    print_header("RECENT SIGHTINGS")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT person_name, timestamp, camera_id, location, confidence
        FROM sightings
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))
    
    rows = cursor.fetchall()
    
    if not rows:
        print("No sightings recorded yet.")
        return
    
    print(f"\nShowing last {len(rows)} sightings:\n")
    
    for i, row in enumerate(rows, 1):
        vehicle, timestamp, camera, location, conf = row
        print(f"{i}. {vehicle}")
        print(f"   Time: {timestamp}")
        print(f"   Camera: {camera} | Location: {location}")
        print(f"   Confidence: {conf*100:.0f}%")
        print()
    
    conn.close()

def view_statistics():
    """View database statistics"""
    print_header("DATABASE STATISTICS")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Total sightings
    cursor.execute("SELECT COUNT(*) FROM sightings")
    total = cursor.fetchone()[0]
    print(f"\nTotal Sightings: {total}")
    
    # Unique vehicles
    cursor.execute("SELECT COUNT(DISTINCT person_name) FROM sightings")
    unique = cursor.fetchone()[0]
    print(f"Unique Vehicles: {unique}")
    
    # Date range
    cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM sightings")
    min_date, max_date = cursor.fetchone()
    if min_date:
        print(f"First Sighting: {min_date}")
        print(f"Last Sighting: {max_date}")
    
    # Top vehicles
    print("\nTop 10 Most Detected Vehicles:")
    cursor.execute("""
        SELECT person_name, COUNT(*) as count
        FROM sightings
        GROUP BY person_name
        ORDER BY count DESC
        LIMIT 10
    """)
    
    for i, (vehicle, count) in enumerate(cursor.fetchall(), 1):
        print(f"  {i}. {vehicle}: {count} sightings")
    
    conn.close()

def search_vehicle(vehicle_name):
    """Search for specific vehicle"""
    print_header(f"SEARCH: {vehicle_name}")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT timestamp, camera_id, location, confidence
        FROM sightings
        WHERE person_name LIKE ?
        ORDER BY timestamp DESC
    """, (f"%{vehicle_name}%",))
    
    rows = cursor.fetchall()
    
    if not rows:
        print(f"\nNo sightings found for: {vehicle_name}")
        return
    
    print(f"\nFound {len(rows)} sightings:\n")
    
    for i, (timestamp, camera, location, conf) in enumerate(rows, 1):
        print(f"{i}. {timestamp}")
        print(f"   Camera: {camera} | Location: {location}")
        print(f"   Confidence: {conf*100:.0f}%")
        print()
    
    conn.close()

def export_to_csv(filename="vehicle_data.csv"):
    """Export all data to CSV"""
    print_header("EXPORT TO CSV")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT person_name, timestamp, camera_id, location, confidence,
               bbox_x, bbox_y, bbox_w, bbox_h
        FROM sightings
        ORDER BY timestamp DESC
    """)
    
    rows = cursor.fetchall()
    
    if not rows:
        print("No data to export.")
        return
    
    with open(filename, 'w') as f:
        # Header
        f.write("Vehicle,Timestamp,Camera,Location,Confidence,BBox_X,BBox_Y,BBox_W,BBox_H\n")
        
        # Data
        for row in rows:
            f.write(','.join(str(x) for x in row) + '\n')
    
    print(f"\n✓ Exported {len(rows)} records to: {filename}")
    print(f"  File location: {Path(filename).absolute()}")
    
    conn.close()

def view_vehicle_timeline(vehicle_name):
    """View complete timeline for a vehicle"""
    print_header(f"TIMELINE: {vehicle_name}")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT timestamp, camera_id, location
        FROM sightings
        WHERE person_name LIKE ?
        ORDER BY timestamp ASC
    """, (f"%{vehicle_name}%",))
    
    rows = cursor.fetchall()
    
    if not rows:
        print(f"\nNo timeline data for: {vehicle_name}")
        return
    
    print(f"\nComplete Movement Timeline ({len(rows)} locations):\n")
    
    for i, (timestamp, camera, location) in enumerate(rows, 1):
        print(f"Step {i}: {timestamp}")
        print(f"        {camera} → {location}")
        print()
    
    conn.close()

def main_menu():
    """Interactive menu"""
    while True:
        print("\n" + "=" * 70)
        print("  HAWKEYE DATABASE VIEWER")
        print("=" * 70)
        print("\n1. View Recent Sightings")
        print("2. View Statistics")
        print("3. Search Vehicle")
        print("4. View Vehicle Timeline")
        print("5. Export to CSV")
        print("6. Exit")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == '1':
            limit = input("How many recent sightings? (default 50): ").strip()
            limit = int(limit) if limit else 50
            view_all_sightings(limit)
        
        elif choice == '2':
            view_statistics()
        
        elif choice == '3':
            vehicle = input("Enter vehicle name or ID: ").strip()
            if vehicle:
                search_vehicle(vehicle)
        
        elif choice == '4':
            vehicle = input("Enter vehicle name: ").strip()
            if vehicle:
                view_vehicle_timeline(vehicle)
        
        elif choice == '5':
            filename = input("Filename (default: vehicle_data.csv): ").strip()
            filename = filename if filename else "vehicle_data.csv"
            export_to_csv(filename)
        
        elif choice == '6':
            print("\nGoodbye!")
            break
        
        else:
            print("\n❌ Invalid choice")

if __name__ == "__main__":
    # Check if database exists
    if not Path(DB_PATH).exists():
        print(f"❌ Database not found: {DB_PATH}")
        print("   Run the vehicle tracker first to generate data.")
        sys.exit(1)
    
    # If arguments provided, run specific command
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "stats":
            view_statistics()
        elif command == "recent":
            limit = int(sys.argv[2]) if len(sys.argv) > 2 else 50
            view_all_sightings(limit)
        elif command == "search":
            if len(sys.argv) > 2:
                search_vehicle(sys.argv[2])
            else:
                print("Usage: python view_database.py search <vehicle_name>")
        elif command == "export":
            filename = sys.argv[2] if len(sys.argv) > 2 else "vehicle_data.csv"
            export_to_csv(filename)
        elif command == "timeline":
            if len(sys.argv) > 2:
                view_vehicle_timeline(sys.argv[2])
            else:
                print("Usage: python view_database.py timeline <vehicle_name>")
        else:
            print(f"Unknown command: {command}")
            print("Available: stats, recent, search, export, timeline")
    else:
        # Interactive menu
        main_menu()
