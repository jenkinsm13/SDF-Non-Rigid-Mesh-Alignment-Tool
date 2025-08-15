#!/usr/bin/env python3
"""
Validation utilities for SDF Mesh Aligner
"""

import os
import numpy as np
import trimesh
from typing import Tuple, Dict, Any, Optional

def validate_mesh(mesh: trimesh.Trimesh) -> Tuple[bool, str]:
    """
    Validate mesh for alignment
    
    Args:
        mesh: Mesh to validate
        
    Returns:
        (is_valid, error_message)
    """
    # Check basic properties
    if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
        return False, "Mesh has no vertices"
        
    if not hasattr(mesh, 'faces') or len(mesh.faces) == 0:
        return False, "Mesh has no faces"
        
    # Check for degenerate faces
    if mesh.faces.shape[1] != 3:
        return False, "Mesh must be triangular"
        
    # Check for duplicate vertices
    unique_vertices = np.unique(mesh.vertices, axis=0)
    if len(unique_vertices) != len(mesh.vertices):
        return False, "Mesh has duplicate vertices"
        
    # Check for degenerate faces (zero area)
    areas = mesh.area_faces
    if np.any(areas == 0):
        return False, "Mesh has degenerate faces (zero area)"
        
    # Check for manifold mesh
    if not mesh.is_watertight:
        print("Warning: Mesh is not watertight")
        
    return True, "Mesh is valid"

def validate_config(config: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate configuration settings
    
    Args:
        config: Configuration dictionary
        
    Returns:
        (is_valid, error_message)
    """
    try:
        # Validate algorithm parameters
        if 'algorithm' in config:
            alg = config['algorithm']
            
            # Check required parameters
            required_params = ['lambda_smooth', 'lambda_detail', 'lambda_rigid', 
                             'sigma', 'max_iterations', 'tolerance', 'subsample_ratio']
            
            for param in required_params:
                if param not in alg:
                    return False, f"Missing algorithm parameter: {param}"
            
            # Validate parameter ranges
            if not (0.0 <= alg['lambda_smooth'] <= 10.0):
                return False, "lambda_smooth must be between 0.0 and 10.0"
                
            if not (0.0 <= alg['lambda_detail'] <= 10.0):
                return False, "lambda_detail must be between 0.0 and 10.0"
                
            if not (0.0 <= alg['lambda_rigid'] <= 10.0):
                return False, "lambda_rigid must be between 0.0 and 10.0"
                
            if not (0.001 <= alg['sigma'] <= 10.0):
                return False, "sigma must be between 0.001 and 10.0"
                
            if not (1 <= alg['max_iterations'] <= 1000):
                return False, "max_iterations must be between 1 and 1000"
                
            if not (1e-8 <= alg['tolerance'] <= 1e-2):
                return False, "tolerance must be between 1e-8 and 1e-2"
                
            if not (0.01 <= alg['subsample_ratio'] <= 1.0):
                return False, "subsample_ratio must be between 0.01 and 1.0"
        
        # Validate optimization parameters
        if 'optimization' in config:
            opt = config['optimization']
            
            required_opt_params = ['use_gpu', 'use_octree', 'use_sparse', 
                                 'chunk_size', 'cache_size', 'n_threads']
            
            for param in required_opt_params:
                if param not in opt:
                    return False, f"Missing optimization parameter: {param}"
            
            # Validate parameter types and ranges
            if not isinstance(opt['use_gpu'], bool):
                return False, "use_gpu must be a boolean"
                
            if not isinstance(opt['use_octree'], bool):
                return False, "use_octree must be a boolean"
                
            if not isinstance(opt['use_sparse'], bool):
                return False, "use_sparse must be a boolean"
                
            if not (1000 <= opt['chunk_size'] <= 100000):
                return False, "chunk_size must be between 1000 and 100000"
                
            if not (100000 <= opt['cache_size'] <= 10000000):
                return False, "cache_size must be between 100000 and 10000000"
                
            if not (1 <= opt['n_threads'] <= 64):
                return False, "n_threads must be between 1 and 64"
        
        return True, "Configuration is valid"
        
    except Exception as e:
        return False, f"Configuration validation error: {str(e)}"

def validate_file_path(filepath: str, check_exists: bool = True) -> Tuple[bool, str]:
    """
    Validate file path
    
    Args:
        filepath: Path to validate
        check_exists: Whether to check if file exists
        
    Returns:
        (is_valid, error_message)
    """
    if not filepath:
        return False, "File path is empty"
    
    if check_exists and not os.path.exists(filepath):
        return False, f"File does not exist: {filepath}"
    
    # Check file extension
    supported_extensions = ['.ply', '.obj', '.stl', '.off', '.glb', '.gltf']
    file_ext = os.path.splitext(filepath)[1].lower()
    
    if file_ext not in supported_extensions:
        return False, f"Unsupported file format: {file_ext}"
    
    return True, "File path is valid"

def validate_mesh_pair(source_mesh: trimesh.Trimesh, 
                      target_mesh: trimesh.Trimesh) -> Tuple[bool, str]:
    """
    Validate mesh pair for alignment
    
    Args:
        source_mesh: Source mesh
        target_mesh: Target mesh
        
    Returns:
        (is_valid, error_message)
    """
    # Validate individual meshes
    source_valid, source_msg = validate_mesh(source_mesh)
    if not source_valid:
        return False, f"Source mesh validation failed: {source_msg}"
    
    target_valid, target_msg = validate_mesh(target_mesh)
    if not target_valid:
        return False, f"Target mesh validation failed: {target_msg}"
    
    # Check size compatibility
    source_vertices = len(source_mesh.vertices)
    target_vertices = len(target_mesh.vertices)
    
    if source_vertices < 100:
        return False, "Source mesh has too few vertices (< 100)"
    
    if target_vertices < 100:
        return False, "Target mesh has too few vertices (< 100)"
    
    # Check scale compatibility
    source_bounds = source_mesh.bounds
    target_bounds = target_mesh.bounds
    
    source_size = np.max(source_bounds[1] - source_bounds[0])
    target_size = np.max(target_bounds[1] - target_bounds[0])
    
    size_ratio = max(source_size, target_size) / min(source_size, target_size)
    
    if size_ratio > 100:
        return False, f"Meshes have very different scales (ratio: {size_ratio:.1f})"
    
    return True, "Mesh pair is valid for alignment"

def validate_alignment_parameters(params: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate alignment parameters
    
    Args:
        params: Alignment parameters
        
    Returns:
        (is_valid, error_message)
    """
    try:
        # Check required parameters
        required_params = ['lambda_smooth', 'lambda_detail', 'max_iterations']
        
        for param in required_params:
            if param not in params:
                return False, f"Missing required parameter: {param}"
        
        # Validate parameter types and ranges
        if not isinstance(params['lambda_smooth'], (int, float)):
            return False, "lambda_smooth must be a number"
        
        if not isinstance(params['lambda_detail'], (int, float)):
            return False, "lambda_detail must be a number"
        
        if not isinstance(params['max_iterations'], int):
            return False, "max_iterations must be an integer"
        
        # Validate ranges
        if not (0.0 <= params['lambda_smooth'] <= 10.0):
            return False, "lambda_smooth must be between 0.0 and 10.0"
        
        if not (0.0 <= params['lambda_detail'] <= 10.0):
            return False, "lambda_detail must be between 0.0 and 10.0"
        
        if not (1 <= params['max_iterations'] <= 1000):
            return False, "max_iterations must be between 1 and 1000"
        
        return True, "Parameters are valid"
        
    except Exception as e:
        return False, f"Parameter validation error: {str(e)}"

def check_system_requirements() -> Tuple[bool, str]:
    """
    Check if system meets requirements
    
    Returns:
        (meets_requirements, message)
    """
    try:
        import numpy
        import scipy
        import trimesh
        import matplotlib
        
        # Check Python version
        import sys
        if sys.version_info < (3, 8):
            return False, "Python 3.8 or higher is required"
        
        # Check available memory
        import psutil
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        if available_memory_gb < 2.0:
            return False, f"Insufficient memory: {available_memory_gb:.1f} GB available, 2.0 GB required"
        
        # Check CPU cores
        cpu_count = os.cpu_count()
        if cpu_count < 2:
            return False, f"Insufficient CPU cores: {cpu_count} available, 2 required"
        
        return True, f"System meets requirements (Memory: {available_memory_gb:.1f} GB, CPU: {cpu_count} cores)"
        
    except ImportError as e:
        return False, f"Missing required dependency: {str(e)}"
    except Exception as e:
        return False, f"System check error: {str(e)}"

def validate_output_directory(directory: str) -> Tuple[bool, str]:
    """
    Validate output directory
    
    Args:
        directory: Directory path to validate
        
    Returns:
        (is_valid, error_message)
    """
    if not directory:
        return False, "Output directory is empty"
    
    try:
        # Check if directory exists, create if not
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Check if directory is writable
        test_file = os.path.join(directory, "test_write.tmp")
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
        except Exception:
            return False, f"Directory is not writable: {directory}"
        
        return True, "Output directory is valid"
        
    except Exception as e:
        return False, f"Output directory validation error: {str(e)}"
