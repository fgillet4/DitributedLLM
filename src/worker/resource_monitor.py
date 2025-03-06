"""
Resource Monitor for DistributedLLM worker nodes.

Monitors system resources like CPU, RAM, and GPU, providing utilization metrics
to the coordinator for workload balancing decisions.
"""

import logging
import os
import platform
import time
import psutil
from typing import Dict, Any, Optional

# Try importing GPU monitoring libraries
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import gpustat
    GPUSTAT_AVAILABLE = True
except ImportError:
    GPUSTAT_AVAILABLE = False

logger = logging.getLogger(__name__)


class ResourceMonitor:
    """
    Monitors system resources and provides metrics for performance optimization.
    """
    
    def __init__(self, update_interval: float = 1.0):
        """Initialize the resource monitor."""
        self.update_interval = update_interval
        self.last_update = 0
        self.metrics = {}
        
        # Initialize metrics
        self.update()
        
        logger.info(f"Resource monitor initialized. GPU monitoring: {TORCH_AVAILABLE or GPUSTAT_AVAILABLE}")
    
    def update(self) -> Dict[str, Any]:
        """Update resource metrics."""
        current_time = time.time()
        
        # Only update if interval has passed
        if current_time - self.last_update < self.update_interval:
            return self.metrics
        
        self.last_update = current_time
        
        # Update CPU metrics
        self.metrics.update(self._get_cpu_metrics())
        
        # Update memory metrics
        self.metrics.update(self._get_memory_metrics())
        
        # Update disk metrics
        self.metrics.update(self._get_disk_metrics())
        
        # Update GPU metrics if available
        gpu_metrics = self._get_gpu_metrics()
        if gpu_metrics:
            self.metrics.update(gpu_metrics)
        
        return self.metrics
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get the current resource metrics."""
        self.update()
        return self.metrics
    
    def get_cpu_percent(self) -> float:
        """Get the current CPU utilization percentage."""
        self.update()
        return self.metrics.get("cpu_percent", 0.0)
    
    def get_memory_usage(self) -> float:
        """Get the current memory usage in MB."""
        self.update()
        return self.metrics.get("memory_used_mb", 0.0)
    
    def get_available_memory(self) -> float:
        """Get the available memory in MB."""
        self.update()
        return self.metrics.get("memory_available_mb", 0.0)
    
    def get_gpu_memory_usage(self) -> Optional[float]:
        """Get the current GPU memory usage in MB, if available."""
        self.update()
        return self.metrics.get("gpu_memory_used_mb")
    
    def _get_cpu_metrics(self) -> Dict[str, Any]:
        """Get CPU-related metrics."""
        cpu_metrics = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "cpu_count": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "cpu_freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None,
            "cpu_load_1min": os.getloadavg()[0] if hasattr(os, 'getloadavg') else None,
            "cpu_load_5min": os.getloadavg()[1] if hasattr(os, 'getloadavg') else None,
            "cpu_load_15min": os.getloadavg()[2] if hasattr(os, 'getloadavg') else None,
        }
        
        return cpu_metrics
    
    def _get_memory_metrics(self) -> Dict[str, Any]:
        """Get memory-related metrics."""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        memory_metrics = {
            "memory_total_mb": mem.total / (1024 * 1024),
            "memory_available_mb": mem.available / (1024 * 1024),
            "memory_used_mb": mem.used / (1024 * 1024),
            "memory_percent": mem.percent,
            "swap_total_mb": swap.total / (1024 * 1024),
            "swap_used_mb": swap.used / (1024 * 1024),
            "swap_percent": swap.percent
        }
        
        return memory_metrics
    
    def _get_disk_metrics(self) -> Dict[str, Any]:
        """Get disk-related metrics."""
        disk = psutil.disk_usage('/')
        
        disk_metrics = {
            "disk_total_gb": disk.total / (1024 * 1024 * 1024),
            "disk_free_gb": disk.free / (1024 * 1024 * 1024),
            "disk_used_gb": disk.used / (1024 * 1024 * 1024),
            "disk_percent": disk.percent
        }
        
        return disk_metrics
    
    def _get_gpu_metrics(self) -> Optional[Dict[str, Any]]:
        """Get GPU-related metrics if available."""
        if not TORCH_AVAILABLE and not GPUSTAT_AVAILABLE:
            return None
        
        gpu_metrics = {}
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                device_count = torch.cuda.device_count()
                gpu_metrics["gpu_count"] = device_count
                
                if device_count > 0:
                    # Get metrics for the first GPU
                    gpu_metrics["gpu_name"] = torch.cuda.get_device_name(0)
                    
                    # Try to get memory usage
                    if hasattr(torch.cuda, 'memory_allocated') and hasattr(torch.cuda, 'memory_reserved'):
                        allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
                        reserved = torch.cuda.memory_reserved(0) / (1024 * 1024)
                        gpu_metrics["gpu_memory_used_mb"] = allocated
                        gpu_metrics["gpu_memory_reserved_mb"] = reserved
                        
                        # Get total memory
                        if hasattr(torch.cuda, 'get_device_properties'):
                            props = torch.cuda.get_device_properties(0)
                            if hasattr(props, 'total_memory'):
                                total = props.total_memory / (1024 * 1024)
                                gpu_metrics["gpu_memory_total_mb"] = total
                                gpu_metrics["gpu_memory_percent"] = (allocated / total) * 100
            
            except Exception as e:
                logger.warning(f"Error getting PyTorch GPU metrics: {e}")
        
        elif GPUSTAT_AVAILABLE:
            try:
                gpu_stats = gpustat.GPUStatCollection.new_query()
                if gpu_stats.gpus:
                    gpu = gpu_stats.gpus[0]  # First GPU
                    gpu_metrics["gpu_count"] = len(gpu_stats.gpus)
                    gpu_metrics["gpu_name"] = gpu.name
                    gpu_metrics["gpu_temperature"] = gpu.temperature
                    gpu_metrics["gpu_utilization"] = gpu.utilization
                    gpu_metrics["gpu_memory_used_mb"] = gpu.memory_used
                    gpu_metrics["gpu_memory_total_mb"] = gpu.memory_total
                    gpu_metrics["gpu_memory_percent"] = (gpu.memory_used / gpu.memory_total) * 100
            
            except Exception as e:
                logger.warning(f"Error getting gpustat metrics: {e}")
        
        return gpu_metrics


# Simple test to verify functionality
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    monitor = ResourceMonitor()
    
    # Print initial metrics
    print("Initial metrics:")
    for key, value in monitor.get_metrics().items():
        print(f"  {key}: {value}")
    
    # Update and print again
    time.sleep(2)
    print("\nUpdated metrics:")
    for key, value in monitor.get_metrics().items():
        print(f"  {key}: {value}")
    
    # Print specific metrics
    print(f"\nCPU: {monitor.get_cpu_percent()}%")
    print(f"Memory: {monitor.get_memory_usage()} MB")
    print(f"GPU Memory: {monitor.get_gpu_memory_usage()} MB")