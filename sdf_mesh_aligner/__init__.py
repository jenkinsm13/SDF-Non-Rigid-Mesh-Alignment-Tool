#!/usr/bin/env python3
"""
SDF Non-Rigid Mesh Alignment Tool

A Python library for aligning laser scan meshes to photogrammetry meshes
using Signed Distance Function (SDF) based energy minimization.

Features:
- Non-rigid mesh alignment with detail preservation
- Multi-resolution optimization
- GPU acceleration support
- Sparse matrix operations for large meshes
- Hierarchical control point selection
- Automatic outlier rejection
"""

__version__ = "2.0.0"
__author__ = "Advanced Mesh Processing"
__email__ = "contact@meshprocessing.com"

# Core classes
from .core.aligner import SDFMeshAligner, OptimizedSDFMeshAligner
from .core.mesh_utils import MeshUtils
from .core.point_cloud_processor import PointCloudProcessor
from .core.hybrid_aligner import HybridAligner
from .gui.app import MeshAlignmentGUI
from .gui.hybrid_gui import HybridAlignmentGUI
from .config.settings import ConfigManager

# Utility functions
from .utils.performance import PerformanceMonitor
from .utils.validation import validate_mesh, validate_config

# Version info
__all__ = [
    "SDFMeshAligner",
    "OptimizedSDFMeshAligner", 
    "MeshUtils",
    "PointCloudProcessor",
    "HybridAligner",
    "MeshAlignmentGUI",
    "HybridAlignmentGUI",
    "ConfigManager",
    "PerformanceMonitor",
    "validate_mesh",
    "validate_config",
    "__version__",
    "__author__",
    "__email__",
]

# Check for GPU support
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Performance info
import os
CPU_CORES = os.cpu_count()

# Print startup info
def print_info():
    """Print library information"""
    print(f"SDF Mesh Aligner v{__version__}")
    print(f"GPU acceleration: {'Available' if GPU_AVAILABLE else 'Not available'}")
    print(f"CPU cores: {CPU_CORES}")
    if not GPU_AVAILABLE:
        print("For GPU acceleration, install CuPy: pip install cupy-cuda11x")

if __name__ == "__main__":
    print_info()
