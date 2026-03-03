"""
HAWKEYE CV-Engine - Complete System with Mini-Backend
Includes face recognition, forensic search, and breadcrumb trails
"""
import cv2
import time
import logging
import numpy as np
import threading
import os
from flask import Flask, Response, request, jsonify, render_template_string
from ultralytics import YOLO

# Import mini-backend
from mini_backend.mini_backend_server import get_mini_backend

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HAWKEYE_COMPLETE")

# Flask app
flask_app = Flask(__name__)
output_frame 