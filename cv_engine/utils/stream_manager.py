"""
Robust Stream Manager for RTSP, HTTP, M3U8, and local video sources
Handles reconnection logic and stream health monitoring
"""
import cv2
import time
import logging
from threading import Thread, Event
from queue import Queue, Empty

logger = logging.getLogger("CV_ENGINE.STREAM")

class StreamManager:
    """
    Manages video stream capture with automatic reconnection
    Supports: RTSP, HTTP, M3U8, local files, and webcams
    """
    
    def __init__(self, source, reconnect_attempts=5, reconnect_delay=3):
        self.source = source
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        
        self.cap = None
        self.is_running = False
        self.frame_queue = Queue(maxsize=2)
        self.stop_event = Event()
        
        self.stream_type = self._detect_stream_type()
        self.connection_failures = 0
        self.last_frame_time = time.time()
        
        logger.info(f"Stream Manager initialized for {self.stream_type}: {source}")
    
    def _detect_stream_type(self):
        """Detect the type of video source"""
        source_str = str(self.source).lower()
        
        if source_str.startswith('rtsp://'):
            return 'RTSP'
        elif source_str.startswith('http://') or source_str.startswith('https://'):
            if source_str.endswith('.m3u8'):
                return 'M3U8'
            return 'HTTP'
        elif source_str.isdigit():
            return 'WEBCAM'
        else:
            return 'FILE'
    
    def _open_stream(self):
        """Open video capture with appropriate settings"""
        logger.info(f"Opening {self.stream_type} stream: {self.source}")
        
        cap = cv2.VideoCapture(self.source)
        
        # Optimize settings for different stream types
        if self.stream_type in ['RTSP', 'HTTP', 'M3U8']:
            # Reduce buffer for live streams to minimize latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Set timeout for network streams (OpenCV 4.5+)
            try:
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            except:
                pass  # Older OpenCV versions don't support these
        
        if not cap.isOpened():
            logger.error(f"Failed to open stream: {self.source}")
            return None
        
        # Verify we can read at least one frame
        ret, frame = cap.read()
        if not ret or frame is None:
            logger.error("Stream opened but cannot read frames")
            cap.release()
            return None
        
        # Put first frame back in queue
        try:
            self.frame_queue.put_nowait((ret, frame))
        except:
            pass
        
        logger.info(f"Stream opened successfully: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        return cap
    
    def _reconnect(self):
        """Attempt to reconnect to the stream"""
        if self.stream_type == 'FILE':
            # For local files, just reopen
            logger.info("Reopening local file")
            return self._open_stream()
        
        # For network streams, implement exponential backoff
        for attempt in range(1, self.reconnect_attempts + 1):
            logger.warning(f"Reconnection attempt {attempt}/{self.reconnect_attempts}")
            
            delay = self.reconnect_delay * attempt  # Exponential backoff
            time.sleep(delay)
            
            cap = self._open_stream()
            if cap is not None:
                self.connection_failures = 0
                logger.info("Reconnection successful")
                return cap
        
        logger.error(f"Failed to reconnect after {self.reconnect_attempts} attempts")
        return None
    
    def _capture_loop(self):
        """Background thread for continuous frame capture"""
        while not self.stop_event.is_set():
            if self.cap is None or not self.cap.isOpened():
                self.cap = self._reconnect()
                if self.cap is None:
                    logger.error("Stream unavailable. Waiting before retry...")
                    time.sleep(10)
                    continue
            
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                self.connection_failures += 1
                logger.warning(f"Frame read failed (failures: {self.connection_failures})")
                
                # For local files, loop back to start
                if self.stream_type == 'FILE':
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                # For network streams, attempt reconnection
                if self.connection_failures >= 3:
                    logger.error("Multiple consecutive failures. Attempting reconnection...")
                    if self.cap:
                        self.cap.release()
                    self.cap = self._reconnect()
                    if self.cap is None:
                        time.sleep(10)
                    continue
            else:
                self.connection_failures = 0
                self.last_frame_time = time.time()
                
                # Update queue (drop old frames for live streams)
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except Empty:
                        pass
                
                try:
                    self.frame_queue.put_nowait((ret, frame))
                except:
                    pass
    
    def start(self):
        """Start the stream manager"""
        if self.is_running:
            logger.warning("Stream manager already running")
            return
        
        self.cap = self._open_stream()
        if self.cap is None:
            raise RuntimeError(f"Failed to open stream: {self.source}")
        
        self.is_running = True
        self.stop_event.clear()
        
        # Start background capture thread
        self.capture_thread = Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        logger.info("Stream manager started")
    
    def read(self):
        """
        Read the next frame from the stream
        Returns: (ret, frame) tuple
        """
        try:
            return self.frame_queue.get(timeout=1.0)
        except Empty:
            logger.warning("Frame queue empty (stream may be stalled)")
            return False, None
    
    def stop(self):
        """Stop the stream manager"""
        logger.info("Stopping stream manager...")
        self.is_running = False
        self.stop_event.set()
        
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
        
        logger.info("Stream manager stopped")
    
    def is_healthy(self):
        """Check if stream is healthy"""
        if not self.is_running:
            return False
        
        # Check if we've received frames recently
        time_since_frame = time.time() - self.last_frame_time
        if time_since_frame > 10:
            logger.warning(f"No frames received for {time_since_frame:.1f}s")
            return False
        
        return True
    
    def get_fps(self):
        """Get stream FPS (if available)"""
        if self.cap:
            return self.cap.get(cv2.CAP_PROP_FPS)
        return 0
    
    def get_resolution(self):
        """Get stream resolution"""
        if self.cap:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        return (0, 0)
