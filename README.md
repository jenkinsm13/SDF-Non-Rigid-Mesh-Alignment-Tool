# SDF Non-Rigid Mesh & Point Cloud Alignment Tool v3.0

A comprehensive Python library for aligning meshes and point clouds using advanced hybrid alignment techniques including Signed Distance Function (SDF) based energy minimization, Coherent Point Drift (CPD), and Fast Point Feature Histograms (FPFH). Optimized for handling large datasets (100k+ points/vertices) efficiently.

## Features

### Core Algorithm
- **Hybrid Alignment System**: Unified framework for mesh-to-mesh, point-to-point, and mixed alignments
- **SDF-based Energy Minimization**: Robust alignment using signed distance functions
- **Coherent Point Drift (CPD)**: Non-rigid point cloud registration
- **FPFH Feature Matching**: Fast Point Feature Histograms for robust initial alignment
- **Non-Rigid Deformation**: Preserves fine surface details while achieving optimal alignment
- **Multi-Resolution Optimization**: Hierarchical approach for better convergence
- **Automatic Outlier Rejection**: Handles noisy data gracefully

### Performance Optimizations
- **GPU Acceleration**: CUDA support via CuPy for massive speedup
- **Sparse Matrix Operations**: Memory-efficient processing for large meshes
- **Octree Spatial Indexing**: Fast proximity queries
- **Parallel Processing**: Multi-threaded optimization
- **Adaptive Sampling**: Intelligent control point selection
- **Memory Management**: Efficient chunked processing

### Advanced Features
- **Hierarchical Control Points**: Multi-level deformation control
- **Curvature-Aware Sampling**: Feature-preserving alignment
- **Automatic Parameter Tuning**: Mesh complexity-based optimization
- **Comprehensive Validation**: Mesh quality and compatibility checks
- **Performance Monitoring**: Real-time metrics and memory tracking

## Installation

### Basic Installation
```bash
pip install -r requirements.txt
```

### With GPU Support (Optional)
```bash
# For CUDA 11.x
pip install cupy-cuda11x

# For CUDA 12.x
pip install cupy-cuda12x
```

### Development Installation
```bash
git clone <repository-url>
cd SDF-Non-Rigid-Mesh-Alignment-Tool
pip install -e .
```

## Quick Start

### GUI Application
```bash
python main.py
# or
python -m sdf_mesh_aligner.gui_app
```

### Command Line Interface
```bash
# Basic mesh alignment
sdf-aligner source.ply target.ply -o aligned.ply

# Point cloud alignment
sdf-aligner source.pcd target.pcd -o aligned.pcd --alignment-method hybrid

# Mixed alignment (mesh to point cloud)
sdf-aligner source.ply target.pcd -o aligned.ply --source-type mesh --target-type pointcloud

# Optimized for large datasets
sdf-aligner source.ply target.ply -o aligned.ply --optimized --gpu

# Custom parameters
sdf-aligner source.ply target.ply -o aligned.ply --lambda-smooth 0.2 --lambda-detail 0.3 --max-iter 100
```

### Python API
```python
from sdf_mesh_aligner import HybridAligner, PointCloudProcessor, MeshUtils
import trimesh

# Load meshes
mesh_utils = MeshUtils()
source_mesh = mesh_utils.load_mesh("source.ply")
target_mesh = mesh_utils.load_mesh("target.ply")

# Create hybrid aligner
aligner = HybridAligner()
aligner.use_gpu = True  # Enable GPU acceleration

# Run mesh-to-mesh alignment
aligned_mesh = aligner.align_hybrid(source_mesh, target_mesh, 'mesh', 'mesh')

# Save result
mesh_utils.save_mesh(aligned_mesh, "aligned.ply")

# Point cloud alignment example
points, normals, colors = PointCloudProcessor.load_point_cloud("source.pcd")
source_pc = {'points': points, 'normals': normals, 'colors': colors}

target_points, target_normals, target_colors = PointCloudProcessor.load_point_cloud("target.pcd")
target_pc = {'points': target_points, 'normals': target_normals, 'colors': target_colors}

# Run point cloud alignment
aligned_points = aligner.align_point_clouds(
    source_pc['points'], target_pc['points'],
    source_pc['normals'], target_pc['normals'],
    method='hybrid'
)
```

## Package Structure

```
sdf_mesh_aligner/
├── __init__.py              # Main package exports
├── core/                    # Core algorithms
│   ├── aligner.py          # SDFMeshAligner, OptimizedSDFMeshAligner
│   ├── hybrid_aligner.py   # HybridAligner for mixed data types
│   ├── point_cloud_processor.py  # PointCloudProcessor utilities
│   └── mesh_utils.py       # Mesh utilities and validation
├── gui/                     # GUI components
│   ├── app.py              # MeshAlignmentGUI (legacy)
│   ├── hybrid_gui.py       # HybridAlignmentGUI (new)
│   └── __init__.py
├── config/                  # Configuration management
│   └── settings.py         # ConfigManager class
├── utils/                   # Utility functions
│   ├── performance.py      # Performance monitoring
│   └── validation.py       # Validation utilities
├── cli/                     # Command line interface
│   └── cli_app.py          # CLI application
└── gui_app.py              # GUI entry point
```

## Configuration

The tool uses JSON-based configuration with automatic validation and preset support.

### Default Configuration
```json
{
  "algorithm": {
    "lambda_smooth": 0.1,
    "lambda_detail": 0.5,
    "lambda_rigid": 0.01,
    "sigma": 0.1,
    "max_iterations": 50,
    "tolerance": 1e-4,
    "subsample_ratio": 0.3
  },
  "optimization": {
    "use_gpu": true,
    "use_octree": true,
    "use_sparse": true,
    "chunk_size": 10000,
    "cache_size": 1000000,
    "n_threads": 8
  }
}
```

### Presets
- **fast**: Quick alignment with reduced accuracy
- **accurate**: High-quality alignment with more iterations
- **memory_efficient**: Optimized for limited memory
- **gpu_optimized**: Maximum GPU utilization

## Algorithm Parameters

### Core Parameters
- **lambda_smooth** (0.001-1.0): Smoothness regularization weight
- **lambda_detail** (0.0-1.0): Detail preservation weight
- **lambda_rigid** (0.001-1.0): Rigidity constraint weight
- **sigma** (0.001-10.0): Gaussian kernel width for RBF interpolation
- **max_iterations** (1-1000): Maximum optimization iterations
- **tolerance** (1e-8-1e-2): Optimization convergence tolerance
- **subsample_ratio** (0.01-1.0): Control point sampling ratio

### Performance Parameters
- **chunk_size** (1000-100000): Processing chunk size for memory management
- **cache_size** (100000-10000000): SDF computation cache size
- **n_threads** (1-64): Number of parallel threads

## Supported Formats

### Input/Output Formats
- **PLY**: Stanford Triangle Format
- **OBJ**: Wavefront Object Format
- **STL**: Stereolithography Format
- **OFF**: Object File Format
- **GLB/GLTF**: glTF Binary/Text Format
- **DAE**: COLLADA Format
- **FBX**: Autodesk FBX Format
- **3DS**: 3D Studio Format

## Performance Guidelines

### Mesh Size Recommendations
- **< 10k vertices**: Standard aligner, default settings
- **10k-100k vertices**: Optimized aligner, enable all optimizations
- **100k-1M vertices**: Reduce control points, increase chunk size
- **> 1M vertices**: Consider mesh decimation first

### Memory Requirements
- **Estimate**: ~1000 bytes per vertex for processing
- **Example**: 100k vertices ≈ 100 MB RAM
- **Large meshes**: Enable sparse matrices and chunked processing

### GPU Acceleration
- **Requirements**: CUDA-capable GPU, CuPy installation
- **Speedup**: 5-20x faster for large meshes
- **Memory**: GPU memory should be 2-4x mesh size

## Examples

### Basic Alignment
```python
from sdf_mesh_aligner import SDFMeshAligner, MeshUtils

# Load meshes
mesh_utils = MeshUtils()
source = mesh_utils.load_mesh("laser_scan.ply")
target = mesh_utils.load_mesh("photogrammetry.ply")

# Create aligner
aligner = SDFMeshAligner()
aligner.lambda_smooth = 0.1
aligner.lambda_detail = 0.5

# Align
aligned, energy = aligner.align(source, target)

# Save
mesh_utils.save_mesh(aligned, "aligned_result.ply")
```

### Large Mesh Optimization
```python
from sdf_mesh_aligner import OptimizedSDFMeshAligner

# Create optimized aligner
aligner = OptimizedSDFMeshAligner()
aligner.use_gpu = True
aligner.use_sparse = True
aligner.chunk_size = 20000
aligner.subsample_ratio = 0.1

# Align large mesh
aligned, energy = aligner.align(large_source, large_target)
```

### Validation and Analysis
```python
from sdf_mesh_aligner import MeshUtils, validate_mesh

# Validate meshes
is_valid, message = validate_mesh(source_mesh)
if not is_valid:
    print(f"Validation failed: {message}")

# Get mesh information
mesh_utils = MeshUtils()
info = mesh_utils.get_mesh_info(source_mesh)
print(f"Vertices: {info['vertices']:,}")
print(f"Faces: {info['faces']:,}")
print(f"Volume: {info['volume']:.6f}")

# Compute alignment metrics
metrics = mesh_utils.compute_alignment_metrics(source, target, aligned)
print(f"Average distance: {metrics['mean_distance']:.6f}")
print(f"Max distance: {metrics['max_distance']:.6f}")
```

## Troubleshooting

### Common Issues
1. **Memory Errors**: Reduce chunk_size, enable sparse matrices
2. **Slow Performance**: Enable GPU acceleration, reduce control points
3. **Poor Alignment**: Adjust lambda parameters, increase iterations
4. **Import Errors**: Install missing dependencies from requirements.txt

### Performance Tips
1. **GPU Acceleration**: Install CuPy for 5-20x speedup
2. **Memory Management**: Close other applications, increase swap space
3. **Mesh Preprocessing**: Remove duplicates, fix normals, decimate if needed
4. **Parameter Tuning**: Use presets or auto-tuning based on mesh size

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{sdf_mesh_aligner,
  title={SDF Non-Rigid Mesh Alignment Tool},
  author={Advanced Mesh Processing},
  year={2024},
  url={https://github.com/your-repo/sdf-mesh-aligner}
}
```

## Acknowledgments

- Trimesh library for mesh processing
- SciPy for optimization algorithms
- NumPy for numerical computations
- Matplotlib for visualization
- CuPy for GPU acceleration
