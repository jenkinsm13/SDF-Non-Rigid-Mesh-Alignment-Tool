#!/usr/bin/env python3
"""
Command Line Interface for SDF Mesh Aligner
"""

import argparse
import sys
import os
import time
import json
from pathlib import Path

from ..core.aligner import SDFMeshAligner, OptimizedSDFMeshAligner
from ..core.mesh_utils import MeshUtils
from ..core.point_cloud_processor import PointCloudProcessor
from ..core.hybrid_aligner import HybridAligner
from ..config.settings import ConfigManager
from ..utils.performance import PerformanceMonitor, performance_monitor
from ..utils.validation import validate_mesh, validate_mesh_pair, check_system_requirements

def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="SDF Non-Rigid Mesh Alignment Tool - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic alignment
  sdf-aligner source.ply target.ply -o aligned.ply
  
  # Optimized alignment for large meshes
  sdf-aligner source.ply target.ply -o aligned.ply --optimized --gpu
  
  # Custom parameters
  sdf-aligner source.ply target.ply -o aligned.ply --lambda-smooth 0.2 --lambda-detail 0.3 --max-iter 100
  
  # Generate report
  sdf-aligner source.ply target.ply -o aligned.ply --report report.txt
  
  # Use preset configuration
  sdf-aligner source.ply target.ply -o aligned.ply --preset fast
        """
    )
    
    # Required arguments
    parser.add_argument("source", help="Source mesh or point cloud file")
    parser.add_argument("target", help="Target mesh or point cloud file")
    parser.add_argument("-o", "--output", required=True, help="Output aligned file")
    
    # Data type options
    parser.add_argument("--source-type", choices=['mesh', 'pointcloud', 'auto'],
                       default='auto', help='Source data type (auto-detect if not specified)')
    parser.add_argument("--target-type", choices=['mesh', 'pointcloud', 'auto'],
                       default='auto', help='Target data type (auto-detect if not specified)')
    parser.add_argument("--alignment-method", choices=['rigid', 'cpd', 'hybrid'],
                       default='hybrid', help='Alignment method (default: hybrid)')
    
    # Algorithm parameters
    parser.add_argument("--lambda-smooth", type=float, default=0.1,
                       help="Smoothness regularization weight (default: 0.1)")
    parser.add_argument("--lambda-detail", type=float, default=0.5,
                       help="Detail preservation weight (default: 0.5)")
    parser.add_argument("--lambda-rigid", type=float, default=0.01,
                       help="Rigidity constraint weight (default: 0.01)")
    parser.add_argument("--sigma", type=float, default=0.1,
                       help="Gaussian kernel width (default: 0.1)")
    parser.add_argument("--max-iter", type=int, default=50,
                       help="Maximum optimization iterations (default: 50)")
    parser.add_argument("--tolerance", type=float, default=1e-4,
                       help="Optimization tolerance (default: 1e-4)")
    parser.add_argument("--subsample-ratio", type=float, default=0.3,
                       help="Control point subsample ratio (default: 0.3)")
    
    # Point cloud specific parameters
    parser.add_argument("--use-fpfh", action="store_true", default=True,
                       help="Use FPFH features for initial alignment (default: True)")
    parser.add_argument("--use-icp-init", action="store_true", default=True,
                       help="Use ICP for initial alignment (default: True)")
    parser.add_argument("--correspondence-dist", type=float, default=0.1,
                       help="Correspondence distance for point clouds (default: 0.1)")
    parser.add_argument("--auto-correspondence", action="store_true", default=True,
                       help="Auto-estimate correspondence distance (default: True)")
    
    # Optimization options
    parser.add_argument("--optimized", action="store_true",
                       help="Use optimized aligner for large meshes")
    parser.add_argument("--gpu", action="store_true",
                       help="Enable GPU acceleration (if available)")
    parser.add_argument("--octree", action="store_true", default=True,
                       help="Use octree spatial indexing (default: True)")
    parser.add_argument("--sparse", action="store_true", default=True,
                       help="Use sparse matrices (default: True)")
    parser.add_argument("--chunk-size", type=int, default=10000,
                       help="Processing chunk size (default: 10000)")
    
    # Output options
    parser.add_argument("--report", help="Generate detailed report file")
    parser.add_argument("--deformation", help="Export deformation field to file")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress progress output")
    
    # Configuration
    parser.add_argument("--config", help="Configuration file")
    parser.add_argument("--preset", choices=["fast", "accurate", "memory_efficient", "gpu_optimized"],
                       help="Use preset configuration")
    
    # Validation
    parser.add_argument("--validate", action="store_true",
                       help="Validate meshes before alignment")
    parser.add_argument("--preprocess", action="store_true",
                       help="Preprocess meshes (remove duplicates, fix normals)")
    
    return parser

def load_data(source_path, target_path, source_type='auto', target_type='auto', 
              preprocess=False, verbose=False):
    """Load and validate meshes or point clouds"""
    
    # Auto-detect data types if not specified
    if source_type == 'auto':
        source_type = detect_data_type(source_path)
    if target_type == 'auto':
        target_type = detect_data_type(target_path)
    
    if verbose:
        print(f"Loading source {source_type}: {source_path}")
    
    try:
        if source_type == 'mesh':
            source_data = MeshUtils.load_mesh(source_path)
            if preprocess:
                source_data = MeshUtils.preprocess_mesh(source_data)
        else:  # pointcloud
            source_data = PointCloudProcessor.load_point_cloud(source_path)
            source_data = {'points': source_data[0], 'normals': source_data[1], 'colors': source_data[2]}
    except Exception as e:
        print(f"Error loading source {source_type}: {e}")
        return None, None, None, None
    
    if verbose:
        print(f"Loading target {target_type}: {target_path}")
    
    try:
        if target_type == 'mesh':
            target_data = MeshUtils.load_mesh(target_path)
            if preprocess:
                target_data = MeshUtils.preprocess_mesh(target_data)
        else:  # pointcloud
            target_data = PointCloudProcessor.load_point_cloud(target_path)
            target_data = {'points': target_data[0], 'normals': target_data[1], 'colors': target_data[2]}
    except Exception as e:
        print(f"Error loading target {target_type}: {e}")
        return None, None, None, None
    
    # Validate data
    if verbose:
        print("Validating data...")
    
    if source_type == 'mesh':
        source_valid, source_msg = validate_mesh(source_data)
        if not source_valid:
            print(f"Source mesh validation failed: {source_msg}")
            return None, None, None, None
    
    if target_type == 'mesh':
        target_valid, target_msg = validate_mesh(target_data)
        if not target_valid:
            print(f"Target mesh validation failed: {target_msg}")
            return None, None, None, None
    
    # Validate pair if both are meshes
    if source_type == 'mesh' and target_type == 'mesh':
        pair_valid, pair_msg = validate_mesh_pair(source_data, target_data)
        if not pair_valid:
            print(f"Mesh pair validation failed: {pair_msg}")
            return None, None, None, None
    
    return source_data, target_data, source_type, target_type

def detect_data_type(file_path):
    """Auto-detect if file is mesh or point cloud based on extension"""
    ext = os.path.splitext(file_path)[1].lower()
    
    # Point cloud formats
    point_cloud_exts = ['.pcd', '.xyz', '.pts', '.txt', '.csv', '.las', '.laz']
    if ext in point_cloud_exts:
        return 'pointcloud'
    
    # Mesh formats (default)
    return 'mesh'

def create_aligner(args, source_data, source_type, verbose=False):
    """Create appropriate aligner based on arguments and data type"""
    
    # Use hybrid aligner for mixed data types or when specified
    if source_type == 'pointcloud' or args.alignment_method in ['rigid', 'cpd', 'hybrid']:
        if verbose:
            print("Using hybrid aligner for point cloud/mixed alignment")
        aligner = HybridAligner()
        
        # Set hybrid aligner parameters
        aligner.lambda_smooth = args.lambda_smooth
        aligner.lambda_detail = args.lambda_detail
        aligner.lambda_rigid = args.lambda_rigid
        aligner.sigma = args.sigma
        aligner.max_iterations = args.max_iter
        aligner.tolerance = args.tolerance
        
        # Point cloud specific parameters
        aligner.use_fpfh = args.use_fpfh
        aligner.use_icp_init = args.use_icp_init
        aligner.correspondence_dist = args.correspondence_dist
        aligner.auto_correspondence = args.auto_correspondence
        
        # Performance parameters
        aligner.use_gpu = args.gpu
        aligner.chunk_size = args.chunk_size
        
    else:
        # Use mesh-specific aligners for mesh-to-mesh alignment
        mesh_utils = MeshUtils()
        
        # Check if optimized aligner should be used
        use_optimized = args.optimized
        if not use_optimized:
            recommendations = mesh_utils.get_recommended_settings(source_data)
            use_optimized = recommendations['use_optimized']
        
        if use_optimized:
            if verbose:
                print("Using optimized aligner for large mesh")
            aligner = OptimizedSDFMeshAligner()
        else:
            if verbose:
                print("Using standard aligner")
            aligner = SDFMeshAligner()
        
        # Set algorithm parameters
        aligner.lambda_smooth = args.lambda_smooth
        aligner.lambda_detail = args.lambda_detail
        aligner.lambda_rigid = args.lambda_rigid
        aligner.sigma = args.sigma
        aligner.max_iterations = args.max_iter
        aligner.tolerance = args.tolerance
        aligner.subsample_ratio = args.subsample_ratio
        
        # Set optimization parameters for optimized aligner
        if hasattr(aligner, 'use_gpu'):
            aligner.use_gpu = args.gpu
            aligner.use_octree = args.octree
        aligner.use_sparse = args.sparse
        aligner.chunk_size = args.chunk_size
    
    return aligner

def progress_callback(message, progress, quiet=False):
    """Progress callback for alignment"""
    if not quiet:
        print(f"[{progress*100:.1f}%] {message}")

def save_results(aligned_data, output_path, source_type='mesh', deformation_path=None, verbose=False):
    """Save alignment results"""
    
    # Save aligned data
    if verbose:
        print(f"Saving aligned {source_type} to: {output_path}")
    
    try:
        if source_type == 'mesh' or isinstance(aligned_data, trimesh.Trimesh):
            # Save mesh
            success = MeshUtils.save_mesh(aligned_data, output_path)
            if not success:
                print("Error: Failed to save aligned mesh")
                return False
        else:
            # Save point cloud
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(aligned_data['points'])
            if aligned_data.get('normals') is not None:
                pcd.normals = o3d.utility.Vector3dVector(aligned_data['normals'])
            if aligned_data.get('colors') is not None:
                pcd.colors = o3d.utility.Vector3dVector(aligned_data['colors'])
            o3d.io.write_point_cloud(output_path, pcd)
            
            if verbose:
                print(f"Saved {len(aligned_data['points']):,} points")
    
    except Exception as e:
        print(f"Error: Failed to save aligned {source_type}: {e}")
        return False
    
    # Save deformation field if requested (only for meshes)
    if deformation_path and source_type == 'mesh':
        if verbose:
            print(f"Saving deformation field to: {deformation_path}")
        
        try:
            import numpy as np
            # Note: This would need source_mesh to be available
            # deformation = aligned_data.vertices - source_mesh.vertices
            # np.savez(deformation_path, deformation=deformation)
            print("Warning: Deformation field export not implemented in CLI")
        except Exception as e:
            print(f"Error saving deformation field: {e}")
    
    return True

def generate_report(source_mesh, target_mesh, aligned_mesh, report_path, 
                   performance_monitor, elapsed_time, verbose=False):
    """Generate detailed alignment report"""
    if verbose:
        print(f"Generating report: {report_path}")
    
    try:
        mesh_utils = MeshUtils()
        
        # Compute alignment metrics
        metrics = mesh_utils.compute_alignment_metrics(source_mesh, target_mesh, aligned_mesh)
        
        # Get performance summary
        perf_summary = performance_monitor.get_performance_summary()
        
        # Write report
        with open(report_path, 'w') as f:
            f.write("SDF MESH ALIGNMENT REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Mesh information
            f.write("MESH INFORMATION:\n")
            f.write(f"  Source vertices: {len(source_mesh.vertices):,}\n")
            f.write(f"  Source faces: {len(source_mesh.faces):,}\n")
            f.write(f"  Target vertices: {len(target_mesh.vertices):,}\n")
            f.write(f"  Target faces: {len(target_mesh.faces):,}\n")
            f.write(f"  Aligned vertices: {len(aligned_mesh.vertices):,}\n")
            f.write(f"  Aligned faces: {len(aligned_mesh.faces):,}\n\n")
            
            # Alignment metrics
            f.write("ALIGNMENT METRICS:\n")
            for key, value in metrics.items():
                f.write(f"  {key}: {value:.6f}\n")
            f.write("\n")
            
            # Performance information
            f.write("PERFORMANCE:\n")
            f.write(f"  Total time: {elapsed_time:.2f} seconds\n")
            f.write(f"  Memory usage: {perf_summary['memory_usage']['rss_mb']:.1f} MB\n")
            f.write(f"  CPU cores: {perf_summary['system_info']['cpu_count']}\n")
            f.write(f"  Available memory: {perf_summary['system_info']['available_memory_gb']:.1f} GB\n")
        
        return True
        
    except Exception as e:
        print(f"Error generating report: {e}")
        return False

def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Check system requirements
    if not args.quiet:
        print("SDF Mesh Aligner v2.0 - Command Line Interface")
        print("="*50)
    
    meets_requirements, req_message = check_system_requirements()
    if not meets_requirements:
        print(f"System requirements not met: {req_message}")
        sys.exit(1)
    
    if not args.quiet:
        print(f"System: {req_message}")
        print("="*50)
    
    # Load configuration if specified
    config_manager = None
    if args.config:
        config_manager = ConfigManager(args.config)
        if not args.quiet:
            print(f"Loaded configuration: {args.config}")
    
    # Apply preset if specified
    if args.preset and config_manager:
        config_manager.apply_preset(args.preset)
        if not args.quiet:
            print(f"Applied preset: {args.preset}")
    
    # Load data
    source_data, target_data, source_type, target_type = load_data(
        args.source, args.target, 
        source_type=args.source_type,
        target_type=args.target_type,
        preprocess=args.preprocess, 
        verbose=args.verbose
    )
    
    if source_data is None or target_data is None:
        sys.exit(1)
    
    # Create aligner
    aligner = create_aligner(args, source_data, source_type, verbose=args.verbose)
    
    # Run alignment
    if not args.quiet:
        print(f"Starting {source_type}-to-{target_type} alignment...")
    
    start_time = time.time()
    
    with performance_monitor("alignment"):
        try:
            if isinstance(aligner, HybridAligner):
                # Use hybrid alignment
                aligned_data = aligner.align_hybrid(
                    source_data, 
                    target_data,
                    source_type,
                    target_type,
                    progress_callback=lambda msg, prog: progress_callback(msg, prog, args.quiet)
                )
                final_energy = 0.0  # Hybrid aligner doesn't return energy
            else:
                # Use mesh-specific alignment
                aligned_data, final_energy = aligner.align(
                    source_data, 
                    target_data,
                    progress_callback=lambda msg, prog: progress_callback(msg, prog, args.quiet)
                )
        except Exception as e:
            print(f"Alignment failed: {e}")
            sys.exit(1)
    
    elapsed_time = time.time() - start_time
    
    # Save results
    success = save_results(
        aligned_data, 
        args.output, 
        source_type=source_type,
        deformation_path=args.deformation,
        verbose=args.verbose
    )
    
    if not success:
        sys.exit(1)
    
    # Generate report if requested
    if args.report:
        # Get performance monitor from context manager
        # This is a simplified version - in practice you'd need to pass the monitor
        performance_monitor = PerformanceMonitor()
        generate_report(
            source_mesh, target_mesh, aligned_mesh, args.report,
            performance_monitor, elapsed_time, verbose=args.verbose
        )
    
    # Print summary
    if not args.quiet:
        print("\n" + "="*50)
        print("ALIGNMENT COMPLETE")
        print("="*50)
        print(f"Time: {elapsed_time:.2f} seconds")
        print(f"Final energy: {final_energy:.6f}")
        print(f"Output: {args.output}")
        
        # Compute and display metrics
        if source_type == 'mesh' and target_type == 'mesh':
            mesh_utils = MeshUtils()
            metrics = mesh_utils.compute_alignment_metrics(source_data, target_data, aligned_data)
            print(f"Average distance to target: {metrics['mean_distance']:.6f}")
            print(f"Maximum distance to target: {metrics['max_distance']:.6f}")
        else:
            print(f"Aligned {source_type} with {len(aligned_data['points'] if source_type == 'pointcloud' else aligned_data.vertices):,} points/vertices")
        print("="*50)

if __name__ == "__main__":
    main()
