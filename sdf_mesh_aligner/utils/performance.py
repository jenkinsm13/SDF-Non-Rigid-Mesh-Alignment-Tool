#!/usr/bin/env python3
"""
Performance monitoring utilities for SDF Mesh Aligner
"""

import time
import psutil
import os
import threading
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager

class PerformanceMonitor:
    """Monitor and track performance metrics"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.metrics = {}
        self.process = psutil.Process(os.getpid())
        
    def start_monitoring(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        self.metrics = {
            'cpu_percent': [],
            'memory_mb': [],
            'disk_io': [],
            'timestamps': []
        }
        
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.end_time = time.time()
        
    def record_metric(self, metric_name: str, value: Any):
        """Record a custom metric"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)
        
    def get_current_stats(self) -> Dict[str, float]:
        """Get current system statistics"""
        try:
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            return {
                'cpu_percent': cpu_percent,
                'memory_mb': memory_mb,
                'memory_gb': memory_mb / 1024,
                'timestamp': time.time()
            }
        except Exception:
            return {
                'cpu_percent': 0.0,
                'memory_mb': 0.0,
                'memory_gb': 0.0,
                'timestamp': time.time()
            }
    
    def get_total_time(self) -> float:
        """Get total elapsed time"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return time.time() - self.start_time
        return 0.0
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics"""
        try:
            memory_info = self.process.memory_info()
            return {
                'rss_mb': memory_info.rss / (1024 * 1024),
                'vms_mb': memory_info.vms / (1024 * 1024),
                'percent': self.process.memory_percent()
            }
        except Exception:
            return {
                'rss_mb': 0.0,
                'vms_mb': 0.0,
                'percent': 0.0
            }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        try:
            return {
                'cpu_count': os.cpu_count(),
                'total_memory_gb': psutil.virtual_memory().total / (1024**3),
                'available_memory_gb': psutil.virtual_memory().available / (1024**3),
                'platform': os.name,
                'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
            }
        except Exception:
            return {
                'cpu_count': 1,
                'total_memory_gb': 0.0,
                'available_memory_gb': 0.0,
                'platform': 'unknown',
                'python_version': 'unknown'
            }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        summary = {
            'total_time': self.get_total_time(),
            'current_stats': self.get_current_stats(),
            'memory_usage': self.get_memory_usage(),
            'system_info': self.get_system_info(),
            'custom_metrics': {}
        }
        
        # Add custom metrics
        for key, values in self.metrics.items():
            if key not in ['cpu_percent', 'memory_mb', 'disk_io', 'timestamps']:
                if values:
                    if isinstance(values[0], (int, float)):
                        summary['custom_metrics'][key] = {
                            'min': min(values),
                            'max': max(values),
                            'mean': sum(values) / len(values),
                            'count': len(values)
                        }
                    else:
                        summary['custom_metrics'][key] = values
        
        return summary
    
    def print_summary(self):
        """Print performance summary to console"""
        summary = self.get_performance_summary()
        
        print("\n" + "="*50)
        print("PERFORMANCE SUMMARY")
        print("="*50)
        print(f"Total time: {summary['total_time']:.2f} seconds")
        print(f"Memory usage: {summary['memory_usage']['rss_mb']:.1f} MB")
        print(f"CPU cores: {summary['system_info']['cpu_count']}")
        print(f"Available memory: {summary['system_info']['available_memory_gb']:.1f} GB")
        
        if summary['custom_metrics']:
            print("\nCustom Metrics:")
            for key, value in summary['custom_metrics'].items():
                if isinstance(value, dict):
                    print(f"  {key}: min={value['min']:.3f}, max={value['max']:.3f}, mean={value['mean']:.3f}")
                else:
                    print(f"  {key}: {value}")
        
        print("="*50)

@contextmanager
def performance_monitor(name: str = "operation"):
    """Context manager for performance monitoring"""
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    try:
        yield monitor
    finally:
        monitor.stop_monitoring()
        print(f"\nPerformance for {name}:")
        monitor.print_summary()

class MemoryTracker:
    """Track memory usage over time"""
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.monitor = PerformanceMonitor()
        self.tracking = False
        self.track_thread = None
        
    def start_tracking(self):
        """Start memory tracking in background thread"""
        if not self.tracking:
            self.tracking = True
            self.monitor.start_monitoring()
            self.track_thread = threading.Thread(target=self._track_loop)
            self.track_thread.daemon = True
            self.track_thread.start()
    
    def stop_tracking(self):
        """Stop memory tracking"""
        self.tracking = False
        if self.track_thread:
            self.track_thread.join()
        self.monitor.stop_monitoring()
    
    def _track_loop(self):
        """Background tracking loop"""
        while self.tracking:
            stats = self.monitor.get_current_stats()
            self.monitor.record_metric('memory_mb', stats['memory_mb'])
            self.monitor.record_metric('cpu_percent', stats['cpu_percent'])
            self.monitor.record_metric('timestamps', stats['timestamp'])
            time.sleep(self.interval)
    
    def get_memory_history(self) -> Dict[str, list]:
        """Get memory usage history"""
        return {
            'memory_mb': self.monitor.metrics.get('memory_mb', []),
            'cpu_percent': self.monitor.metrics.get('cpu_percent', []),
            'timestamps': self.monitor.metrics.get('timestamps', [])
        }

def estimate_memory_requirement(n_vertices: int, n_faces: int) -> float:
    """Estimate memory requirement for mesh processing"""
    # Rough estimation based on typical mesh processing
    vertex_memory = n_vertices * 24  # 3 floats * 8 bytes per vertex
    face_memory = n_faces * 12       # 3 ints * 4 bytes per face
    adjacency_memory = n_vertices * 50  # Adjacency data structures
    processing_overhead = 2.0  # 2x overhead for processing
    
    total_mb = (vertex_memory + face_memory + adjacency_memory) * processing_overhead / (1024 * 1024)
    return total_mb

def check_memory_available(required_mb: float) -> bool:
    """Check if required memory is available"""
    try:
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        return available_mb >= required_mb
    except Exception:
        return True  # Assume available if can't check

def optimize_for_memory(mesh_vertices: int, mesh_faces: int) -> Dict[str, Any]:
    """Get optimization recommendations based on memory constraints"""
    estimated_memory = estimate_memory_requirement(mesh_vertices, mesh_faces)
    available_memory = psutil.virtual_memory().available / (1024 * 1024)
    
    recommendations = {
        'use_sparse_matrices': True,
        'chunk_size': 5000,
        'subsample_ratio': 0.1,
        'max_iterations': 30,
        'use_octree': True,
        'cache_size': 500000
    }
    
    if estimated_memory > available_memory * 0.5:
        # High memory usage - aggressive optimization
        recommendations.update({
            'chunk_size': 2000,
            'subsample_ratio': 0.05,
            'max_iterations': 20,
            'cache_size': 200000
        })
    elif estimated_memory > available_memory * 0.25:
        # Medium memory usage - moderate optimization
        recommendations.update({
            'chunk_size': 5000,
            'subsample_ratio': 0.1,
            'max_iterations': 30,
            'cache_size': 500000
        })
    else:
        # Low memory usage - minimal optimization
        recommendations.update({
            'chunk_size': 10000,
            'subsample_ratio': 0.2,
            'max_iterations': 50,
            'cache_size': 1000000
        })
    
    return recommendations
