"""
Vector Store for Face Embeddings
Uses in-memory storage with optional persistence
"""
import numpy as np
import json
import os
from datetime import datetime
import logging

logger = logging.getLogger("MINI_BACKEND.VECTOR_STORE")

class VectorStore:
    """Simple vector store for face embeddings with cosine similarity search"""
    
    def __init__(self, storage_file="vector_store.json"):
        self.storage_file = storage_file
        self.vectors = {}  # {person_id: {"name": str, "embedding": list, "sightings": []}}
        self.next_id = 1
        self.load()
    
    def add_person(self, name, embedding):
        """Add a new person with their face embedding"""
        person_id = f"person_{self.next_id}"
        self.next_id += 1
        
        self.vectors[person_id] = {
            "name": name,
            "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
            "sightings": [],
            "created_at": datetime.now().isoformat()
        }
        
        self.save()
        logger.info(f"✓ Added person: {name} (ID: {person_id})")
        return person_id
    
    def add_sighting(self, person_id, location, timestamp, confidence, camera_id=None):
        """Add a sighting for a person"""
        if person_id not in self.vectors:
            return False
        
        sighting = {
            "timestamp": timestamp,
            "location": location,
            "confidence": confidence,
            "camera_id": camera_id
        }
        
        self.vectors[person_id]["sightings"].append(sighting)
        self.save()
        return True
    
    def search(self, query_embedding, threshold=0.4, top_k=5):
        """Search for similar faces using cosine similarity"""
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding)
        
        results = []
        
        for person_id, data in self.vectors.items():
            stored_embedding = np.array(data["embedding"])
            
            # Cosine similarity
            similarity = np.dot(query_embedding, stored_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
            )
            
            if similarity >= threshold:
                results.append({
                    "person_id": person_id,
                    "name": data["name"],
                    "similarity": float(similarity),
                    "sightings_count": len(data["sightings"])
                })
        
        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]
    
    def get_person(self, person_id):
        """Get person details"""
        return self.vectors.get(person_id)
    
    def get_all_persons(self):
        """Get all registered persons"""
        return [
            {
                "person_id": pid,
                "name": data["name"],
                "sightings_count": len(data["sightings"]),
                "created_at": data.get("created_at")
            }
            for pid, data in self.vectors.items()
        ]
    
    def get_breadcrumb_trail(self, person_id):
        """Get chronological movement trail for a person"""
        if person_id not in self.vectors:
            return []
        
        sightings = self.vectors[person_id]["sightings"]
        # Sort by timestamp
        sorted_sightings = sorted(sightings, key=lambda x: x["timestamp"])
        return sorted_sightings
    
    def save(self):
        """Save to disk"""
        try:
            with open(self.storage_file, 'w') as f:
                json.dump({
                    "vectors": self.vectors,
                    "next_id": self.next_id
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
    
    def load(self):
        """Load from disk"""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)
                    self.vectors = data.get("vectors", {})
                    self.next_id = data.get("next_id", 1)
                logger.info(f"✓ Loaded {len(self.vectors)} persons from storage")
            except Exception as e:
                logger.error(f"Failed to load vector store: {e}")
    
    def clear(self):
        """Clear all data"""
        self.vectors = {}
        self.next_id = 1
        self.save()
        logger.info("✓ Vector store cleared")
