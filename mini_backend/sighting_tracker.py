"""
Sighting Tracker - Stores and retrieves person sightings
Provides breadcrumb trail and forensic search capabilities
"""
import sqlite3
import json
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger("MINI_BACKEND.TRACKER")

class SightingTracker:
    """
    Tracks person sightings with SQLite
    Provides forensic search and breadcrumb trail features
    """
    
    def __init__(self, db_path="mini_backend/sightings.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        self._create_tables()
        logger.info("SightingTracker initialized")
    
    def _create_tables(self):
        """Create database tables"""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sightings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_name TEXT NOT NULL,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                camera_id TEXT,
                location TEXT,
                bbox_x INTEGER,
                bbox_y INTEGER,
                bbox_w INTEGER,
                bbox_h INTEGER,
                metadata TEXT
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_person_name 
            ON sightings(person_name)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON sightings(timestamp)
        """)
        
        self.conn.commit()
        cursor.close()
    
    def add_sighting(self, person_name, confidence, camera_id="CAM-01", 
                    location="Unknown", bbox=None, metadata=None):
        """Add a new sighting"""
        cursor = self.conn.cursor()
        bbox_x, bbox_y, bbox_w, bbox_h = bbox if bbox else (0, 0, 0, 0)
        metadata_json = json.dumps(metadata) if metadata else "{}"
        
        cursor.execute("""
            INSERT INTO sightings 
            (person_name, confidence, camera_id, location, 
             bbox_x, bbox_y, bbox_w, bbox_h, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (person_name, confidence, camera_id, location,
              bbox_x, bbox_y, bbox_w, bbox_h, metadata_json))
        
        self.conn.commit()
        
        sighting_id = cursor.lastrowid
        cursor.close()
        logger.debug(f"Sighting added: {person_name} (ID: {sighting_id})")
        return sighting_id
    
    def get_breadcrumb_trail(self, person_name, limit=100):
        """
        Get chronological breadcrumb trail for a person
        Returns list of sightings ordered by time
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM sightings
            WHERE person_name = ?
            ORDER BY timestamp ASC
            LIMIT ?
        """, (person_name, limit))
        
        rows = cursor.fetchall()
        cursor.close()
        
        trail = []
        for row in rows:
            trail.append({
                'id': row['id'],
                'person_name': row['person_name'],
                'confidence': row['confidence'],
                'timestamp': row['timestamp'],
                'camera_id': row['camera_id'],
                'location': row['location'],
                'bbox': {
                    'x': row['bbox_x'],
                    'y': row['bbox_y'],
                    'w': row['bbox_w'],
                    'h': row['bbox_h']
                },
                'metadata': json.loads(row['metadata']) if row['metadata'] else {}
            })
        
        logger.info(f"Breadcrumb trail for {person_name}: {len(trail)} sightings")
        return trail
    
    def forensic_search(self, person_name=None, start_time=None, end_time=None, 
                       camera_id=None, limit=100):
        """
        Forensic search with multiple filters
        """
        cursor = self.conn.cursor()
        query = "SELECT * FROM sightings WHERE 1=1"
        params = []
        
        if person_name:
            query += " AND person_name = ?"
            params.append(person_name)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        if camera_id:
            query += " AND camera_id = ?"
            params.append(camera_id)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        cursor.close()
        
        results = []
        for row in rows:
            results.append({
                'id': row['id'],
                'person_name': row['person_name'],
                'confidence': row['confidence'],
                'timestamp': row['timestamp'],
                'camera_id': row['camera_id'],
                'location': row['location'],
                'bbox': {
                    'x': row['bbox_x'],
                    'y': row['bbox_y'],
                    'w': row['bbox_w'],
                    'h': row['bbox_h']
                }
            })
        
        logger.info(f"Forensic search: {len(results)} results")
        return results
    
    def get_recent_sightings(self, limit=50):
        """Get most recent sightings"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM sightings
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        cursor.close()
        
        sightings = []
        for row in rows:
            sightings.append({
                'id': row['id'],
                'person_name': row['person_name'],
                'confidence': row['confidence'],
                'timestamp': row['timestamp'],
                'camera_id': row['camera_id'],
                'location': row['location']
            })
        
        return sightings
    
    def get_statistics(self):
        """Get database statistics"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) as total FROM sightings")
        total = cursor.fetchone()['total']
        
        cursor.execute("""
            SELECT person_name, COUNT(*) as count 
            FROM sightings 
            GROUP BY person_name 
            ORDER BY count DESC
        """)
        by_person = [dict(row) for row in cursor.fetchall()]
        cursor.close()
        
        return {
            'total_sightings': total,
            'by_person': by_person
        }
    
    def clear_old_sightings(self, days=30):
        """Clear sightings older than specified days"""
        cursor = self.conn.cursor()
        cursor.execute("""
            DELETE FROM sightings
            WHERE timestamp < datetime('now', '-' || ? || ' days')
        """, (days,))
        
        deleted = cursor.rowcount
        self.conn.commit()
        cursor.close()
        
        logger.info(f"Cleared {deleted} old sightings")
        return deleted
    
    def close(self):
        """Close database connection"""
        self.conn.close()
