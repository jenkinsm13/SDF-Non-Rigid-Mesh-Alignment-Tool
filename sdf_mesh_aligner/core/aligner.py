#!/usr/bin/env python3
"""
Core SDF-based mesh alignment algorithms
Combines original and optimized implementations with advanced features
"""

import sys
import numpy as np
import trimesh
import scipy.sparse as sp
import scipy.optimize as optimize
from scipy.spatial import cKDTree
from scipy.sparse.linalg import spsolve, cg
from scipy.ndimage import distance_transform_edt
import time
import os
import psutil
import warnings
warnings.filterwarnings('ignore')

# Try to import optional GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = np  # Fallback to numpy

# Try to import Numba for JIT compilation
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Define fallback decorators
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

class SDFMeshAligner:
    """Original SDF-based mesh alignment algorithm"""
    
    def __init__(self):
        # Algorithm parameters
        self.lambda_smooth = 0.1      # Smoothness regularization
        self.lambda_detail = 0.5      # Detail preservation weight
        self.lambda_rigid = 0.01      # Rigidity constraint
        self.sigma = 0.1               # Gaussian kernel width
        self.max_iterations = 50
        self.tolerance = 1e-4
        self.subsample_ratio = 0.3    # For control points
        
    def compute_sdf(self, mesh, query_points, cache_size=1000):
        """Compute signed distance function values"""
        # Use proximity queries for efficiency
        proximity = trimesh.proximity.ProximityQuery(mesh)
        
        # Compute signed distances in batches
        n_points = len(query_points)
        sdf_values = np.zeros(n_points)
        
        for i in range(0, n_points, cache_size):
            batch = query_points[i:min(i+cache_size, n_points)]
            sdf_values[i:min(i+cache_size, n_points)] = proximity.signed_distance(batch)
            
        return sdf_values
    
    def select_control_points(self, mesh, subsample_ratio):
        """Select control points using feature-aware sampling"""
        n_vertices = len(mesh.vertices)
        n_control = max(100, int(n_vertices * subsample_ratio))
        
        # Compute vertex importance based on curvature
        if hasattr(mesh, 'vertex_defects'):
            vertex_defects = mesh.vertex_defects
        else:
            # Simple approximation using vertex normals variation
            vertex_defects = np.ones(n_vertices)
            
        # Importance sampling
        importance = np.abs(vertex_defects) + 0.1
        importance = importance / importance.sum()
        
        # Sample control points
        control_indices = np.random.choice(
            n_vertices, size=n_control, replace=False, p=importance
        )
        
        return control_indices
    
    def build_deformation_graph(self, vertices, control_indices):
        """Build deformation graph with RBF interpolation"""
        control_vertices = vertices[control_indices]
        n_vertices = len(vertices)
        n_control = len(control_indices)
        
        # Build interpolation matrix
        W = np.zeros((n_vertices, n_control))
        
        # Compute RBF weights
        for i, v in enumerate(vertices):
            distances = np.linalg.norm(control_vertices - v, axis=1)
            weights = np.exp(-distances**2 / (2 * self.sigma**2))
            weights = weights / (weights.sum() + 1e-10)
            W[i, :] = weights
            
        return W
    
    def energy_function(self, deformation_params, source_vertices, target_mesh, 
                       control_indices, W, initial_vertices):
        """Energy function for optimization"""
        n_control = len(control_indices)
        
        # Reshape deformation parameters
        deformation = deformation_params.reshape(n_control, 3)
        
        # Apply deformation
        vertex_deformation = W @ deformation
        deformed_vertices = source_vertices + vertex_deformation
        
        # Data term: SDF matching energy
        source_sdf = self.compute_sdf(
            trimesh.Trimesh(vertices=deformed_vertices, faces=source_vertices), 
            deformed_vertices[:100]  # Sample for efficiency
        )
        target_sdf = self.compute_sdf(target_mesh, deformed_vertices[:100])
        data_energy = np.sum((source_sdf - target_sdf)**2)
        
        # Smoothness term
        laplacian = np.sum(np.diff(deformation, axis=0)**2)
        smooth_energy = self.lambda_smooth * laplacian
        
        # Detail preservation term
        detail_loss = np.sum((deformed_vertices - initial_vertices)**2)
        detail_energy = self.lambda_detail * detail_loss
        
        # Rigidity term (preserve local structure)
        rigid_energy = 0
        for i in range(min(10, n_control-1)):  # Sample edges
            edge_original = source_vertices[control_indices[i+1]] - source_vertices[control_indices[i]]
            edge_deformed = (source_vertices[control_indices[i+1]] + vertex_deformation[control_indices[i+1]]) - \
                          (source_vertices[control_indices[i]] + vertex_deformation[control_indices[i]])
            rigid_energy += np.sum((edge_deformed - edge_original)**2)
        rigid_energy *= self.lambda_rigid
        
        total_energy = data_energy + smooth_energy + detail_energy + rigid_energy
        return total_energy
    
    def align(self, source_mesh, target_mesh, progress_callback=None):
        """Main alignment function"""
        print("Starting SDF-based mesh alignment...")
        
        # Select control points
        if progress_callback:
            progress_callback("Selecting control points...", 0.1)
        control_indices = self.select_control_points(source_mesh, self.subsample_ratio)
        print(f"Selected {len(control_indices)} control points")
        
        # Build deformation graph
        if progress_callback:
            progress_callback("Building deformation graph...", 0.2)
        W = self.build_deformation_graph(source_mesh.vertices, control_indices)
        
        # Initialize deformation parameters
        n_control = len(control_indices)
        deformation_params = np.zeros(n_control * 3)
        
        # Multi-resolution optimization
        resolutions = [1.0, 0.5, 0.25]
        best_params = deformation_params.copy()
        best_energy = float('inf')
        
        for res_idx, resolution in enumerate(resolutions):
            if progress_callback:
                progress_callback(f"Optimizing at resolution {resolution:.2f}...", 
                                0.3 + 0.5 * res_idx / len(resolutions))
            
            self.sigma = 0.2 * resolution
            
            # Subsample for efficiency at lower resolutions
            if resolution < 1.0:
                sample_size = int(len(source_mesh.vertices) * resolution)
                sample_indices = np.random.choice(
                    len(source_mesh.vertices), 
                    size=sample_size, 
                    replace=False
                )
                source_vertices = source_mesh.vertices[sample_indices]
            else:
                source_vertices = source_mesh.vertices
            
            # Optimize at this resolution
            result = optimize.minimize(
                self.energy_function,
                deformation_params,
                args=(source_vertices, target_mesh, control_indices, W, source_mesh.vertices),
                method='L-BFGS-B',
                options={
                    'maxiter': self.max_iterations,
                    'ftol': self.tolerance,
                    'disp': True
                }
            )
            
            if result.fun < best_energy:
                best_energy = result.fun
                best_params = result.x
                
            deformation_params = result.x
            
            print(f"Resolution {resolution}: Energy = {result.fun:.6f}")
        
        # Apply final deformation
        if progress_callback:
            progress_callback("Applying final deformation...", 0.9)
            
        final_deformation = best_params.reshape(n_control, 3)
        vertex_deformation = W @ final_deformation
        aligned_vertices = source_mesh.vertices + vertex_deformation
        
        # Create aligned mesh
        aligned_mesh = trimesh.Trimesh(
            vertices=aligned_vertices,
            faces=source_mesh.faces,
            vertex_normals=source_mesh.vertex_normals
        )
        
        if progress_callback:
            progress_callback("Alignment complete!", 1.0)
            
        return aligned_mesh, best_energy


class OptimizedSDFMeshAligner(SDFMeshAligner):
    """Optimized alignment algorithm for large meshes with advanced features"""
    
    def __init__(self):
        super().__init__()
        
        # Optimization parameters
        self.use_gpu = GPU_AVAILABLE
        self.use_octree = True
        self.use_sparse = True
        self.chunk_size = 10000       # Process vertices in chunks
        self.cache_size = 1000000     # SDF cache size
        self.n_threads = os.cpu_count()
        
        # Memory management
        self.max_memory_gb = psutil.virtual_memory().available / (1024**3) * 0.5
        
    @staticmethod
    @jit(nopython=True, parallel=True)
    def compute_distances_batch(points, reference_points):
        """Optimized distance computation using Numba"""
        n_points = points.shape[0]
        n_ref = reference_points.shape[0]
        distances = np.zeros(n_points)
        indices = np.zeros(n_points, dtype=np.int32)
        
        for i in prange(n_points):
            min_dist = np.inf
            min_idx = 0
            for j in range(n_ref):
                dx = points[i, 0] - reference_points[j, 0]
                dy = points[i, 1] - reference_points[j, 1]
                dz = points[i, 2] - reference_points[j, 2]
                dist = np.sqrt(dx*dx + dy*dy + dz*dz)
                if dist < min_dist:
                    min_dist = dist
                    min_idx = j
            distances[i] = min_dist
            indices[i] = min_idx
            
        return distances, indices
    
    def build_octree(self, mesh):
        """Build octree for efficient spatial queries"""
        if not self.use_octree:
            return None
            
        # Use trimesh's built-in spatial indexing
        return mesh.kdtree
    
    def compute_sdf_optimized(self, mesh, query_points, octree=None, use_cache=True):
        """Optimized SDF computation with caching and batching"""
        n_points = len(query_points)
        sdf_values = np.zeros(n_points)
        
        # Use GPU if available
        if self.use_gpu and GPU_AVAILABLE:
            query_points_gpu = cp.asarray(query_points)
            mesh_vertices_gpu = cp.asarray(mesh.vertices)
            
            # Process in chunks to manage GPU memory
            for i in range(0, n_points, self.chunk_size):
                end_idx = min(i + self.chunk_size, n_points)
                chunk = query_points_gpu[i:end_idx]
                
                # Compute distances on GPU
                distances = cp.min(cp.linalg.norm(
                    chunk[:, None, :] - mesh_vertices_gpu[None, :, :], 
                    axis=2
                ), axis=1)
                
                sdf_values[i:end_idx] = cp.asnumpy(distances)
        else:
            # CPU implementation with octree
            if octree is not None:
                distances, indices = octree.query(query_points, k=1)
                sdf_values = distances
            else:
                # Fallback to proximity query
                proximity = trimesh.proximity.ProximityQuery(mesh)
                
                # Process in chunks for memory efficiency
                for i in range(0, n_points, self.chunk_size):
                    end_idx = min(i + self.chunk_size, n_points)
                    batch = query_points[i:end_idx]
                    sdf_values[i:end_idx] = proximity.signed_distance(batch)
                    
        return sdf_values
    
    def hierarchical_control_points(self, mesh, base_ratio=0.1):
        """Select control points using hierarchical clustering"""
        n_vertices = len(mesh.vertices)
        
        # Compute vertex importance using multiple features
        features = []
        
        # 1. Curvature-based importance
        if hasattr(mesh, 'vertex_defects'):
            curvature = np.abs(mesh.vertex_defects)
        else:
            # Approximate curvature using normal variation
            curvature = self.estimate_curvature(mesh)
        features.append(curvature / (curvature.max() + 1e-10))
        
        # 2. Edge length variation
        edge_lengths = mesh.edges_unique_length
        vertex_edge_var = np.zeros(n_vertices)
        for edge, length in zip(mesh.edges, edge_lengths):
            vertex_edge_var[edge[0]] += length
            vertex_edge_var[edge[1]] += length
        features.append(vertex_edge_var / (vertex_edge_var.max() + 1e-10))
        
        # Combined importance
        importance = np.mean(features, axis=0) + 0.1
        importance = importance / importance.sum()
        
        # Hierarchical selection
        levels = 3
        control_indices = []
        
        for level in range(levels):
            level_ratio = base_ratio * (2 ** level)
            n_control = min(int(n_vertices * level_ratio), n_vertices)
            
            # Weighted sampling without replacement
            indices = np.random.choice(
                n_vertices, 
                size=n_control, 
                replace=False, 
                p=importance
            )
            control_indices.append(indices)
            
        return control_indices
    
    def estimate_curvature(self, mesh):
        """Fast curvature estimation"""
        n_vertices = len(mesh.vertices)
        curvature = np.zeros(n_vertices)
        
        # Use vertex normal variation as proxy for curvature
        vertex_faces = mesh.vertex_faces
        
        for i in range(n_vertices):
            faces = vertex_faces[i]
            faces = faces[faces != -1]  # Remove invalid faces
            
            if len(faces) > 1:
                normals = mesh.face_normals[faces]
                # Compute variation in normals
                mean_normal = normals.mean(axis=0)
                curvature[i] = np.mean(np.linalg.norm(normals - mean_normal, axis=1))
                
        return curvature
    
    def build_sparse_deformation_matrix(self, vertices, control_indices, bandwidth=0.1):
        """Build sparse deformation matrix for memory efficiency"""
        n_vertices = len(vertices)
        n_control = len(control_indices)
        control_vertices = vertices[control_indices]
        
        # Build sparse matrix
        rows = []
        cols = []
        data = []
        
        # Use KD-tree for efficient neighbor queries
        tree = cKDTree(control_vertices)
        
        # Adaptive bandwidth based on mesh density
        distances, _ = tree.query(vertices, k=min(10, n_control))
        local_bandwidth = np.median(distances, axis=1) * bandwidth
        
        for i in range(n_vertices):
            # Find nearby control points
            nearby_indices = tree.query_ball_point(vertices[i], local_bandwidth[i] * 3)
            
            if len(nearby_indices) == 0:
                # Fallback to k-nearest
                _, nearby_indices = tree.query(vertices[i], k=min(5, n_control))
                
            # Compute RBF weights
            nearby_controls = control_vertices[nearby_indices]
            distances = np.linalg.norm(nearby_controls - vertices[i], axis=1)
            weights = np.exp(-distances**2 / (2 * local_bandwidth[i]**2))
            weights = weights / (weights.sum() + 1e-10)
            
            # Add to sparse matrix
            for j, idx in enumerate(nearby_indices):
                if weights[j] > 1e-6:  # Threshold for sparsity
                    rows.append(i)
                    cols.append(idx)
                    data.append(weights[j])
                    
        # Create sparse matrix
        W = sp.csr_matrix((data, (rows, cols)), shape=(n_vertices, n_control))
        
        print(f"Sparsity: {100 * (1 - len(data) / (n_vertices * n_control)):.2f}%")
        
        return W
    
    def compute_laplacian_matrix(self, mesh, control_indices):
        """Compute sparse Laplacian matrix for smoothness regularization"""
        n_control = len(control_indices)
        
        # Map vertex indices to control indices
        vertex_to_control = {v: i for i, v in enumerate(control_indices)}
        
        # Build adjacency for control points
        rows = []
        cols = []
        data = []
        
        edges = mesh.edges
        for edge in edges:
            if edge[0] in vertex_to_control and edge[1] in vertex_to_control:
                i = vertex_to_control[edge[0]]
                j = vertex_to_control[edge[1]]
                
                # Add Laplacian entries
                rows.extend([i, j, i, j])
                cols.extend([j, i, i, j])
                data.extend([1, 1, -1, -1])
                
        # Create sparse Laplacian
        L = sp.csr_matrix((data, (rows, cols)), shape=(n_control, n_control))
        
        return L
    
    def parallel_energy_computation(self, deformation, source_vertices, target_mesh,
                                   W, L, control_indices, octree=None):
        """Parallelized energy computation"""
        n_control = len(control_indices)
        deformation = deformation.reshape(n_control, 3)
        
        # Apply deformation using sparse matrix
        vertex_deformation = W @ deformation
        deformed_vertices = source_vertices + vertex_deformation
        
        # Sample points for SDF computation (adaptive sampling)
        n_samples = min(5000, len(source_vertices))
        sample_indices = np.random.choice(len(source_vertices), n_samples, replace=False)
        sampled_deformed = deformed_vertices[sample_indices]
        
        # Compute data term using optimized SDF
        target_sdf = self.compute_sdf_optimized(target_mesh, sampled_deformed, octree)
        data_energy = np.mean(target_sdf**2)
        
        # Smoothness term using sparse Laplacian
        if self.use_sparse:
            smooth_x = deformation[:, 0].T @ L @ deformation[:, 0]
            smooth_y = deformation[:, 1].T @ L @ deformation[:, 1]
            smooth_z = deformation[:, 2].T @ L @ deformation[:, 2]
            smooth_energy = self.lambda_smooth * (smooth_x + smooth_y + smooth_z)
        else:
            laplacian = np.sum(np.diff(deformation, axis=0)**2)
            smooth_energy = self.lambda_smooth * laplacian
        
        # Detail preservation (L2 regularization)
        detail_energy = self.lambda_detail * np.sum(vertex_deformation**2)
        
        total_energy = data_energy + smooth_energy + detail_energy
        
        return total_energy
    
    def optimize_with_lbfgs(self, initial_params, args, maxiter=50):
        """Memory-efficient L-BFGS optimization"""
        # Use scipy's L-BFGS with limited memory
        result = optimize.minimize(
            self.parallel_energy_computation,
            initial_params,
            args=args,
            method='L-BFGS-B',
            options={
                'maxiter': maxiter,
                'ftol': self.tolerance,
                'gtol': 1e-5,
                'maxls': 20,
                'maxcor': 10,  # Limited memory
                'disp': True
            }
        )
        return result
    
    def align(self, source_mesh, target_mesh, progress_callback=None):
        """Optimized alignment for large meshes"""
        print(f"Starting optimized alignment for {len(source_mesh.vertices)} vertices...")
        print(f"Available memory: {self.max_memory_gb:.2f} GB")
        print(f"GPU acceleration: {'Enabled' if self.use_gpu else 'Disabled'}")
        
        # Build spatial index for target mesh
        if progress_callback:
            progress_callback("Building spatial index...", 0.05)
        target_octree = self.build_octree(target_mesh)
        
        # Hierarchical control point selection
        if progress_callback:
            progress_callback("Selecting control points...", 0.1)
        control_levels = self.hierarchical_control_points(source_mesh)
        
        # Multi-resolution optimization
        resolutions = [(0.25, control_levels[0]), 
                      (0.5, control_levels[1]), 
                      (1.0, control_levels[2])]
        
        best_deformation = None
        best_energy = float('inf')
        
        for res_idx, (resolution, control_indices) in enumerate(resolutions):
            if progress_callback:
                progress_callback(f"Optimizing at resolution {resolution:.2f}...", 
                                0.2 + 0.6 * res_idx / len(resolutions))
            
            print(f"\nResolution {resolution}: {len(control_indices)} control points")
            
            # Subsample vertices for this resolution
            if resolution < 1.0:
                n_sample = int(len(source_mesh.vertices) * resolution)
                sample_indices = np.random.choice(
                    len(source_mesh.vertices), 
                    n_sample, 
                    replace=False
                )
                vertices_subset = source_mesh.vertices[sample_indices]
            else:
                vertices_subset = source_mesh.vertices
                sample_indices = np.arange(len(source_mesh.vertices))
            
            # Build sparse deformation matrix
            W = self.build_sparse_deformation_matrix(
                vertices_subset, 
                control_indices[sample_indices] if resolution < 1.0 else control_indices,
                bandwidth=self.sigma * (2 - resolution)
            )
            
            # Build Laplacian for regularization
            L = self.compute_laplacian_matrix(source_mesh, control_indices)
            
            # Initialize deformation
            n_control = len(control_indices)
            if best_deformation is not None and len(best_deformation) == n_control * 3:
                initial_deformation = best_deformation
            else:
                initial_deformation = np.zeros(n_control * 3)
            
            # Optimize
            result = self.optimize_with_lbfgs(
                initial_deformation,
                args=(vertices_subset, target_mesh, W, L, control_indices, target_octree),
                maxiter=int(self.max_iterations * resolution)
            )
            
            if result.fun < best_energy:
                best_energy = result.fun
                best_deformation = result.x
                best_W = W
                best_control_indices = control_indices
                
            print(f"Energy: {result.fun:.6f}")
        
        # Apply final deformation to full mesh
        if progress_callback:
            progress_callback("Applying final deformation...", 0.9)
            
        # Build full deformation matrix if needed
        if len(best_control_indices) < len(source_mesh.vertices):
            W_full = self.build_sparse_deformation_matrix(
                source_mesh.vertices, 
                best_control_indices,
                bandwidth=self.sigma
            )
        else:
            W_full = best_W
            
        final_deformation = best_deformation.reshape(len(best_control_indices), 3)
        vertex_deformation = W_full @ final_deformation
        aligned_vertices = source_mesh.vertices + vertex_deformation
        
        # Create aligned mesh
        aligned_mesh = trimesh.Trimesh(
            vertices=aligned_vertices,
            faces=source_mesh.faces,
            vertex_normals=source_mesh.vertex_normals if hasattr(source_mesh, 'vertex_normals') else None
        )
        
        # Clear GPU memory if used
        if self.use_gpu and GPU_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()
            
        if progress_callback:
            progress_callback("Alignment complete!", 1.0)
            
        return aligned_mesh, best_energy
