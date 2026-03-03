"""
VRAM Monitor for GPU memory management
Prevents OOM crashes during high-resolution image enhancement
"""
import logging
import time
from threading import Thread, Event

logger = logging.getLogger("CV_ENGINE.VRAM")

class VRAMMonitor:
    """
    Monitors GPU memory usage and provides warnings/throttling
    to prevent Out-Of-Memory crashes
    """
    
    def __init__(self, limit_mb=4096, warning_threshold=0.85, critical_threshold=0.95):
        self.limit_mb = limit_mb
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        
        self.is_monitoring = False
        self.stop_event = Event()
        self.has_cuda = False
        
        # Try to import torch for CUDA monitoring
        try:
            import torch
            if torch.cuda.is_available():
                self.torch = torch
                self.has_cuda = True
                self.device_count = torch.cuda.device_count()
                logger.info(f"VRAM Monitor initialized: {self.device_count} CUDA device(s) detected")
            else:
                logger.info("VRAM Monitor: CUDA not available, monitoring disabled")
        except ImportError:
            logger.info("VRAM Monitor: PyTorch not installed, monitoring disabled")
    
    def get_memory_stats(self, device_id=0):
        """
        Get current memory statistics for a GPU device
        
        Returns:
            dict with allocated, reserved, and free memory in MB
        """
        if not self.has_cuda:
            return None
        
        try:
            allocated = self.torch.cuda.memory_allocated(device_id) / (1024 ** 2)
            reserved = self.torch.cuda.memory_reserved(device_id) / (1024 ** 2)
            total = self.torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 2)
            free = total - allocated
            
            return {
                'allocated_mb': allocated,
                'reserved_mb': reserved,
                'total_mb': total,
                'free_mb': free,
                'usage_percent': (allocated / total) * 100
            }
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return None
    
    def check_memory_status(self, device_id=0):
        """
        Check current memory status and return warning level
        
        Returns:
            'ok', 'warning', or 'critical'
        """
        stats = self.get_memory_stats(device_id)
        if stats is None:
            return 'ok'
        
        usage_ratio = stats['allocated_mb'] / stats['total_mb']
        
        if usage_ratio >= self.critical_threshold:
            return 'critical'
        elif usage_ratio >= self.warning_threshold:
            return 'warning'
        else:
            return 'ok'
    
    def can_allocate(self, required_mb, device_id=0):
        """
        Check if we can safely allocate the requested memory
        
        Args:
            required_mb: Memory required in MB
            device_id: GPU device ID
        
        Returns:
            bool: True if allocation is safe
        """
        if not self.has_cuda:
            return True  # No monitoring, assume OK
        
        stats = self.get_memory_stats(device_id)
        if stats is None:
            return True
        
        # Check if allocation would exceed limit
        projected_usage = stats['allocated_mb'] + required_mb
        
        if projected_usage > self.limit_mb:
            logger.warning(f"Allocation denied: {required_mb}MB would exceed limit ({projected_usage}/{self.limit_mb}MB)")
            return False
        
        # Check if allocation would exceed critical threshold
        projected_ratio = projected_usage / stats['total_mb']
        if projected_ratio > self.critical_threshold:
            logger.warning(f"Allocation denied: Would reach critical threshold ({projected_ratio*100:.1f}%)")
            return False
        
        return True
    
    def clear_cache(self, device_id=0):
        """Clear CUDA cache to free up memory"""
        if not self.has_cuda:
            return
        
        try:
            self.torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while not self.stop_event.is_set():
            for device_id in range(self.device_count):
                status = self.check_memory_status(device_id)
                stats = self.get_memory_stats(device_id)
                
                if stats:
                    if status == 'critical':
                        logger.error(f"GPU {device_id} CRITICAL: {stats['usage_percent']:.1f}% used ({stats['allocated_mb']:.0f}/{stats['total_mb']:.0f}MB)")
                        self.clear_cache(device_id)
                    elif status == 'warning':
                        logger.warning(f"GPU {device_id} WARNING: {stats['usage_percent']:.1f}% used ({stats['allocated_mb']:.0f}/{stats['total_mb']:.0f}MB)")
            
            time.sleep(5)  # Check every 5 seconds
    
    def start_monitoring(self):
        """Start background monitoring thread"""
        if not self.has_cuda:
            logger.info("VRAM monitoring not started (CUDA unavailable)")
            return
        
        if self.is_monitoring:
            logger.warning("VRAM monitoring already running")
            return
        
        self.is_monitoring = True
        self.stop_event.clear()
        
        self.monitor_thread = Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("VRAM monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        if not self.is_monitoring:
            return
        
        logger.info("Stopping VRAM monitoring...")
        self.is_monitoring = False
        self.stop_event.set()
        
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2.0)
        
        logger.info("VRAM monitoring stopped")
    
    def get_summary(self):
        """Get summary of all GPU memory usage"""
        if not self.has_cuda:
            return "VRAM Monitoring: CUDA not available"
        
        summary = []
        for device_id in range(self.device_count):
            stats = self.get_memory_stats(device_id)
            if stats:
                device_name = self.torch.cuda.get_device_name(device_id)
                summary.append(
                    f"GPU {device_id} ({device_name}): "
                    f"{stats['allocated_mb']:.0f}/{stats['total_mb']:.0f}MB "
                    f"({stats['usage_percent']:.1f}%)"
                )
        
        return "\n".join(summary) if summary else "No GPU stats available"
