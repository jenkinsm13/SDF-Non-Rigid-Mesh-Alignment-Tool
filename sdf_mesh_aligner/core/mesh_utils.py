#!/usr/bin/env python3
"""
Mesh utilities for loading, validation, and analysis
"""

import os
import numpy as np
import trimesh
from typing import Optional, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class MeshUtils:
    """Utility class for mesh operations"""
    
    SUPPORTED_FORMATS = {
        '.ply': 'PLY',
        '.obj': 'OBJ', 
        '.stl': 'STL',
        '.off': 'OFF',
        '.glb': 'GLB',
        '.gltf': 'GLTF',
        '.dae': 'DAE',
        '.fbx': 'FBX',
        '.3ds': '3DS'
    }
    
    def __init__(self):
        self.loaded_meshes = {}
        
    @staticmethod
    def load_mesh(filename: str, force_mesh: bool = True) -> trimesh.Trimesh:
        """
        Load mesh from file with error handling
        
        Args:
            filename: Path to mesh file
            force_mesh: Force conversion to single mesh
            
        Returns:
            Loaded mesh object
            
        Raises:
            ValueError: If file cannot be loaded
        """
        if not os.path.exists(filename):
            raise ValueError(f"File not found: {filename}")
            
        try:
            # Load mesh
            mesh = trimesh.load(filename, force='mesh' if force_mesh else None)
            
            # Handle scene objects
            if isinstance(mesh, trimesh.Scene):
                meshes = list(mesh.geometry.values())
                if len(meshes) == 1:
                    mesh = meshes[0]
                else:
                    # Combine multiple meshes
                    mesh = trimesh.util.concatenate(meshes)
                    
            # Validate mesh
            if not isinstance(mesh, trimesh.Trimesh):
                raise ValueError(f"Failed to load mesh from {filename}")
                
            # Ensure mesh has required attributes
            if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
                raise ValueError(f"Mesh has no vertices: {filename}")
                
            if not hasattr(mesh, 'faces') or len(mesh.faces) == 0:
                raise ValueError(f"Mesh has no faces: {filename}")
                
            return mesh
            
        except Exception as e:
            raise ValueError(f"Failed to load mesh from {filename}: {str(e)}")
    
    @staticmethod
    def save_mesh(mesh: trimesh.Trimesh, filename: str) -> bool:
        """
        Save mesh to file
        
        Args:
            mesh: Mesh to save
            filename: Output filename
            
        Returns:
            True if successful
        """
        try:
            mesh.export(filename)
            return True
        except Exception as e:
            print(f"Failed to save mesh to {filename}: {str(e)}")
            return False
    
    @staticmethod
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
    
    @staticmethod
    def get_mesh_info(mesh: trimesh.Trimesh) -> Dict[str, Any]:
        """
        Get comprehensive mesh information
        
        Args:
            mesh: Mesh to analyze
            
        Returns:
            Dictionary with mesh information
        """
        info = {
            'vertices': len(mesh.vertices),
            'faces': len(mesh.faces),
            'edges': len(mesh.edges),
            'bounds': mesh.bounds.tolist(),
            'volume': float(mesh.volume),
            'area': float(mesh.area),
            'is_watertight': mesh.is_watertight,
            'is_winding_consistent': mesh.is_winding_consistent,
            'is_empty': mesh.is_empty,
            'is_valid': mesh.is_valid,
        }
        
        # Add curvature information if available
        if hasattr(mesh, 'vertex_defects'):
            info['curvature_available'] = True
            info['mean_curvature'] = float(np.mean(np.abs(mesh.vertex_defects)))
        else:
            info['curvature_available'] = False
            
        # Add bounding box dimensions
        bounds = mesh.bounds
        info['dimensions'] = (bounds[1] - bounds[0]).tolist()
        
        return info
    
    @staticmethod
    def preprocess_mesh(mesh: trimesh.Trimesh, 
                       remove_duplicates: bool = True,
                       fix_normals: bool = True,
                       remove_degenerate: bool = True) -> trimesh.Trimesh:
        """
        Preprocess mesh for alignment
        
        Args:
            mesh: Mesh to preprocess
            remove_duplicates: Remove duplicate vertices
            fix_normals: Fix face normals
            remove_degenerate: Remove degenerate faces
            
        Returns:
            Preprocessed mesh
        """
        processed_mesh = mesh.copy()
        
        if remove_duplicates:
            # Remove duplicate vertices
            unique_vertices, inverse_indices = np.unique(
                processed_mesh.vertices, axis=0, return_inverse=True
            )
            processed_mesh.vertices = unique_vertices
            processed_mesh.faces = inverse_indices[processed_mesh.faces]
            
        if remove_degenerate:
            # Remove degenerate faces
            areas = processed_mesh.area_faces
            valid_faces = areas > 1e-10
            processed_mesh.faces = processed_mesh.faces[valid_faces]
            
        if fix_normals:
            # Fix face normals
            processed_mesh.fix_normals()
            
        return processed_mesh
    
    @staticmethod
    def compute_alignment_metrics(source_mesh: trimesh.Trimesh, 
                                target_mesh: trimesh.Trimesh,
                                aligned_mesh: trimesh.Trimesh) -> Dict[str, float]:
        """
        Compute alignment quality metrics
        
        Args:
            source_mesh: Original source mesh
            target_mesh: Target mesh
            aligned_mesh: Aligned mesh
            
        Returns:
            Dictionary with alignment metrics
        """
        # Compute distances to target
        proximity = trimesh.proximity.ProximityQuery(target_mesh)
        
        # Sample points for evaluation
        n_samples = min(10000, len(aligned_mesh.vertices))
        sample_indices = np.random.choice(
            len(aligned_mesh.vertices), n_samples, replace=False
        )
        sample_vertices = aligned_mesh.vertices[sample_indices]
        
        # Compute signed distances
        distances = proximity.signed_distance(sample_vertices)
        
        metrics = {
            'mean_distance': float(np.mean(np.abs(distances))),
            'max_distance': float(np.max(np.abs(distances))),
            'std_distance': float(np.std(distances)),
            'median_distance': float(np.median(np.abs(distances))),
            'rms_distance': float(np.sqrt(np.mean(distances**2))),
            'percentile_95': float(np.percentile(np.abs(distances), 95)),
            'percentile_99': float(np.percentile(np.abs(distances), 99)),
        }
        
        # Compute volume change
        original_volume = source_mesh.volume
        aligned_volume = aligned_mesh.volume
        metrics['volume_change_ratio'] = float(aligned_volume / original_volume)
        
        # Compute surface area change
        original_area = source_mesh.area
        aligned_area = aligned_mesh.area
        metrics['area_change_ratio'] = float(aligned_area / original_area)
        
        return metrics
    
    @staticmethod
    def decimate_mesh(mesh: trimesh.Trimesh, 
                     target_vertices: Optional[int] = None,
                     target_faces: Optional[int] = None,
                     ratio: float = 0.5) -> trimesh.Trimesh:
        """
        Decimate mesh to reduce complexity
        
        Args:
            mesh: Mesh to decimate
            target_vertices: Target number of vertices
            target_faces: Target number of faces
            ratio: Reduction ratio (0.0 to 1.0)
            
        Returns:
            Decimated mesh
        """
        if target_vertices is not None:
            # Decimate to target vertex count
            decimated = mesh.simplify_quadratic_decimation(target_vertices)
        elif target_faces is not None:
            # Decimate to target face count
            decimated = mesh.simplify_quadratic_decimation(target_faces)
        else:
            # Decimate by ratio
            target_faces = int(len(mesh.faces) * ratio)
            decimated = mesh.simplify_quadratic_decimation(target_faces)
            
        return decimated
    
    @staticmethod
    def normalize_mesh(mesh: trimesh.Trimesh, 
                      scale_to_unit: bool = True,
                      center_to_origin: bool = True) -> trimesh.Trimesh:
        """
        Normalize mesh (scale and center)
        
        Args:
            mesh: Mesh to normalize
            scale_to_unit: Scale to unit bounding box
            center_to_origin: Center at origin
            
        Returns:
            Normalized mesh
        """
        normalized_mesh = mesh.copy()
        
        if center_to_origin:
            # Center at origin
            centroid = normalized_mesh.centroid
            normalized_mesh.vertices -= centroid
            
        if scale_to_unit:
            # Scale to unit bounding box
            bounds = normalized_mesh.bounds
            scale_factor = 1.0 / np.max(bounds[1] - bounds[0])
            normalized_mesh.vertices *= scale_factor
            
        return normalized_mesh
    
    @staticmethod
    def estimate_mesh_complexity(mesh: trimesh.Trimesh) -> str:
        """
        Estimate mesh complexity level
        
        Args:
            mesh: Mesh to analyze
            
        Returns:
            Complexity level string
        """
        n_vertices = len(mesh.vertices)
        
        if n_vertices < 1000:
            return "Low"
        elif n_vertices < 10000:
            return "Medium"
        elif n_vertices < 100000:
            return "High"
        else:
            return "Very High"
    
    @staticmethod
    def get_recommended_settings(mesh: trimesh.Trimesh) -> Dict[str, Any]:
        """
        Get recommended algorithm settings based on mesh complexity
        
        Args:
            mesh: Mesh to analyze
            
        Returns:
            Dictionary with recommended settings
        """
        complexity = MeshUtils.estimate_mesh_complexity(mesh)
        n_vertices = len(mesh.vertices)
        
        if complexity == "Low":
            return {
                'use_optimized': False,
                'subsample_ratio': 0.5,
                'max_iterations': 30,
                'chunk_size': 5000,
                'use_sparse': False,
                'use_octree': False
            }
        elif complexity == "Medium":
            return {
                'use_optimized': True,
                'subsample_ratio': 0.3,
                'max_iterations': 50,
                'chunk_size': 10000,
                'use_sparse': True,
                'use_octree': True
            }
        elif complexity == "High":
            return {
                'use_optimized': True,
                'subsample_ratio': 0.2,
                'max_iterations': 75,
                'chunk_size': 15000,
                'use_sparse': True,
                'use_octree': True
            }
        else:  # Very High
            return {
                'use_optimized': True,
                'subsample_ratio': 0.1,
                'max_iterations': 100,
                'chunk_size': 20000,
                'use_sparse': True,
                'use_octree': True
            }
