#!/usr/bin/env python3
"""
Hybrid aligner for meshes and point clouds
Unified alignment system supporting mesh-to-mesh, point-to-point, and hybrid alignments
"""
import numpy as np
import trimesh
import open3d as o3d
import scipy.sparse as sp
import scipy.optimize as optimize
from scipy.spatial import cKDTree
from scipy.sparse.linalg import spsolve, cg
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from numba import jit, prange
import warnings
warnings.filterwarnings('ignore')

# Try to import optional GPU acceleration
try:
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = np  # Fallback to numpy

from .point_cloud_processor import PointCloudProcessor

class HybridAligner:
    """Unified aligner for meshes and point clouds"""
    
    def __init__(self):
        # Algorithm parameters
        self.lambda_smooth = 0.01  # Lower default
        self.lambda_detail = 0.1   # Lower default for stronger deformation
        self.lambda_rigid = 0.01
        self.sigma = 0.1
        self.max_iterations = 100  # Increased default
        self.tolerance = 1e-4
        
        # Point cloud specific
        self.use_fpfh = True  # Fast Point Feature Histograms
        self.use_icp_init = True  # ICP initialization
        self.correspondence_dist = 0.1
        self.auto_correspondence = True  # Auto-estimate correspondence distance
        
        # Performance parameters
        self.use_gpu = GPU_AVAILABLE
        self.chunk_size = 10000
        self.voxel_size = None  # Auto-compute based on data
        
    def estimate_correspondence_distance(self, source_points, target_points):
        """Automatically estimate appropriate correspondence distance"""
        # Sample points for estimation
        n_sample = min(1000, len(source_points), len(target_points))
        source_sample = source_points[np.random.choice(len(source_points), n_sample, replace=False)]
        target_sample = target_points[np.random.choice(len(target_points), n_sample, replace=False)]
        
        # Compute nearest neighbor distances
        tree = cKDTree(target_sample)
        distances, _ = tree.query(source_sample, k=1)
        
        # Use robust statistics
        median_dist = np.median(distances)
        percentile_95 = np.percentile(distances, 95)
        
        # Estimate based on point cloud characteristics
        estimated_dist = median_dist * 3  # Allow for some deformation
        estimated_dist = min(estimated_dist, percentile_95)  # Cap at 95th percentile
        
        # Also consider point cloud scale
        source_scale = np.linalg.norm(np.max(source_points, axis=0) - np.min(source_points, axis=0))
        target_scale = np.linalg.norm(np.max(target_points, axis=0) - np.min(target_points, axis=0))
        avg_scale = (source_scale + target_scale) / 2
        
        # Correspondence distance should be a small fraction of the overall scale
        estimated_dist = min(estimated_dist, avg_scale * 0.05)
        
        print(f"Auto-estimated correspondence distance: {estimated_dist:.4f}")
        print(f"  (median NN dist: {median_dist:.4f}, scale: {avg_scale:.4f})")
        
        return estimated_dist
        
    def compute_fpfh_features(self, points, normals):
        """Compute Fast Point Feature Histograms for robust matching"""
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # Ensure we have normals
            if normals is None or len(normals) == 0:
                print("Estimating normals for FPFH computation...")
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(30))
                pcd.orient_normals_consistent_tangent_plane(30)
            else:
                pcd.normals = o3d.utility.Vector3dVector(normals)
            
            # Compute bounding box diagonal for radius estimation
            bbox = pcd.get_axis_aligned_bounding_box()
            bbox_diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
            
            # Adaptive radius based on point cloud scale
            radius = max(self.sigma * 5, bbox_diag * 0.05)
            
            print(f"Computing FPFH features with radius={radius:.4f}")
            
            fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                pcd, 
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100)
            )
            
            feature_data = np.asarray(fpfh.data).T
            
            # Validate features
            if feature_data.shape[0] == 0 or np.all(feature_data == 0):
                print("Warning: FPFH features are empty or all zeros")
                # Return random features as fallback
                return np.random.randn(len(points), 33) * 0.1
                
            return feature_data
            
        except Exception as e:
            print(f"Error computing FPFH features: {e}")
            # Return random features as fallback
            return np.random.randn(len(points), 33) * 0.1
    
    def initial_alignment_ransac(self, source_points, target_points, 
                                source_normals, target_normals):
        """RANSAC-based initial alignment using feature matching"""
        try:
            # Downsample if too many points for feature computation
            max_points_for_features = 5000
            if len(source_points) > max_points_for_features:
                print(f"Downsampling source from {len(source_points)} to {max_points_for_features} points for RANSAC")
                indices = np.random.choice(len(source_points), max_points_for_features, replace=False)
                source_points_sample = source_points[indices]
                source_normals_sample = source_normals[indices] if source_normals is not None else None
            else:
                source_points_sample = source_points
                source_normals_sample = source_normals
                
            if len(target_points) > max_points_for_features:
                print(f"Downsampling target from {len(target_points)} to {max_points_for_features} points for RANSAC")
                indices = np.random.choice(len(target_points), max_points_for_features, replace=False)
                target_points_sample = target_points[indices]
                target_normals_sample = target_normals[indices] if target_normals is not None else None
            else:
                target_points_sample = target_points
                target_normals_sample = target_normals
            
            # Compute features
            source_fpfh = self.compute_fpfh_features(source_points_sample, source_normals_sample)
            target_fpfh = self.compute_fpfh_features(target_points_sample, target_normals_sample)
            
            # Create Open3D point clouds
            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(source_points_sample)
            
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(target_points_sample)
            
            # Create feature objects - handling different Open3D versions
            source_feature = o3d.pipelines.registration.Feature()
            source_feature.data = source_fpfh.T
            
            target_feature = o3d.pipelines.registration.Feature()
            target_feature.data = target_fpfh.T
            
            # RANSAC registration with more robust parameters
            result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_pcd, target_pcd,
                source_feature,
                target_feature,
                True,  # mutual_filter
                self.correspondence_dist * 2,  # max_correspondence_distance
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                3,  # ransac_n
                [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                 o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(self.correspondence_dist * 2)],
                o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
            )
            
            print(f"RANSAC alignment: {len(result.correspondence_set)} correspondences found")
            return result.transformation
            
        except Exception as e:
            print(f"RANSAC alignment failed: {e}")
            print("Falling back to identity transform")
            return np.eye(4)
    
    def point_to_point_icp(self, source_points, target_points, init_transform=None):
        """Point-to-point ICP for rigid alignment"""
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source_points)
        
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_points)
        
        if init_transform is None:
            init_transform = np.eye(4)
            
        # Run ICP
        result = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd,
            self.correspondence_dist,
            init_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.max_iterations
            )
        )
        
        return result.transformation, result.fitness
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def compute_point_to_plane_distances(source_points, target_points, target_normals):
        """Fast point-to-plane distance computation"""
        n_source = source_points.shape[0]
        n_target = target_points.shape[0]
        distances = np.zeros(n_source)
        indices = np.zeros(n_source, dtype=np.int32)
        
        for i in prange(n_source):
            min_dist = np.inf
            min_idx = 0
            
            for j in range(n_target):
                # Point-to-point vector
                diff = source_points[i] - target_points[j]
                
                # Project onto normal (point-to-plane distance)
                dist = np.abs(np.dot(diff, target_normals[j]))
                
                if dist < min_dist:
                    min_dist = dist
                    min_idx = j
                    
            distances[i] = min_dist
            indices[i] = min_idx
            
        return distances, indices
    
    def build_deformation_graph_adaptive(self, points, k_neighbors=8):
        """Build adaptive deformation graph for point clouds"""
        n_points = len(points)
        
        # Adjust k_neighbors based on point cloud size
        k_neighbors = min(k_neighbors, n_points - 1, 20)
        k_neighbors = max(3, k_neighbors)  # At least 3 neighbors
        
        # Adaptive sampling based on local density
        try:
            nbrs = NearestNeighbors(n_neighbors=min(k_neighbors+1, n_points), 
                                   algorithm='kd_tree').fit(points)
            distances, _ = nbrs.kneighbors(points)
            
            # Handle edge case where we have fewer points than k_neighbors
            if distances.shape[1] > 1:
                local_density = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-6)
            else:
                local_density = np.ones(n_points)
                
        except Exception as e:
            print(f"Warning: Could not compute local density: {e}")
            local_density = np.ones(n_points)
        
        # Sample control points based on density
        sampling_prob = local_density / (local_density.sum() + 1e-10)
        
        # Determine number of control points
        n_control = min(int(n_points * 0.1), 5000, n_points)  # Max 5000 control points
        n_control = max(10, n_control)  # At least 10 control points if possible
        n_control = min(n_control, n_points)  # Can't have more control points than points
        
        try:
            control_indices = np.random.choice(n_points, n_control, replace=False, p=sampling_prob)
        except Exception as e:
            print(f"Warning: Weighted sampling failed: {e}")
            # Fallback to uniform sampling
            control_indices = np.random.choice(n_points, n_control, replace=False)
        
        # Build graph edges
        control_points = points[control_indices]
        tree = cKDTree(control_points)
        
        # Sparse weight matrix
        rows, cols, data = [], [], []
        
        for i in range(n_points):
            # Find k nearest control points
            k_query = min(k_neighbors, n_control)
            dists, indices = tree.query(points[i], k=k_query)
            
            # Handle single distance case
            if not isinstance(dists, np.ndarray):
                dists = np.array([dists])
                indices = np.array([indices])
            
            # Compute RBF weights
            weights = np.exp(-dists**2 / (2 * self.sigma**2))
            weight_sum = weights.sum()
            if weight_sum > 1e-10:
                weights = weights / weight_sum
            else:
                weights = np.ones_like(weights) / len(weights)
            
            for j, idx in enumerate(indices):
                if weights[j] > 1e-6:
                    rows.append(i)
                    cols.append(idx)
                    data.append(weights[j])
                    
        W = sp.csr_matrix((data, (rows, cols)), shape=(n_points, n_control))
        
        print(f"Deformation graph: {n_points} points, {n_control} control points, "
              f"sparsity: {100 * (1 - len(data) / (n_points * n_control)):.1f}%")
        
        return W, control_indices
    
    def non_rigid_cpd(self, source_points, target_points, w=0.0, max_iter=50):
        """Coherent Point Drift for non-rigid alignment"""
        # Implementation of CPD algorithm
        X = target_points  # Reference
        Y = source_points  # Points to transform
        
        N, D = X.shape
        M, _ = Y.shape
        
        # Initialize parameters
        sigma2 = np.sum((X[None, :, :] - Y[:, None, :])**2) / (D * M * N)
        W = np.zeros((M, D))
        
        for iteration in range(max_iter):
            # E-step: compute responsibilities
            P = np.exp(-np.sum((X[None, :, :] - (Y[:, None, :] + W[:, None, :]))**2, axis=2) / (2 * sigma2))
            P = P / (P.sum(axis=0, keepdims=True) + 1e-10)
            
            # M-step: update transformation
            P1 = P.sum(axis=1)
            Pt1 = P.sum(axis=0)
            
            # Update W (non-rigid transformation)
            if w > 0:
                # Regularized solution
                G = self._gaussian_kernel(Y, Y, self.sigma)
                W = np.linalg.solve(G + w * sigma2 * np.eye(M), P @ X - P1[:, None] * Y)
            else:
                W = (P @ X - P1[:, None] * Y) / P1[:, None]
                
            # Update sigma2
            Yhat = Y + W
            sigma2 = np.sum(P * np.sum((X[None, :, :] - Yhat[:, None, :])**2, axis=2)) / (np.sum(P) * D)
            
            if iteration % 10 == 0:
                print(f"  CPD iteration {iteration}: sigma2 = {sigma2:.6f}")
                
        return Y + W
    
    def align_mesh_to_mesh(self, source_mesh, target_mesh, progress_callback=None):
        """Dedicated mesh-to-mesh alignment preserving topology"""
        print("="*50)
        print("MESH-TO-MESH NON-RIGID ALIGNMENT")
        print(f"Source: {len(source_mesh.vertices)} vertices, {len(source_mesh.faces)} faces")
        print(f"Target: {len(target_mesh.vertices)} vertices, {len(target_mesh.faces)} faces")
        print("="*50)
        
        # Use the specialized mesh aligner from the original tool
        from scipy.spatial import cKDTree
        import scipy.sparse as sp
        
        source_vertices = source_mesh.vertices
        n_vertices = len(source_vertices)
        
        # 1. Select control points based on mesh features
        if progress_callback:
            progress_callback("Selecting control points...", 0.1)
            
        # Use mesh curvature or uniform sampling for control points
        n_control = min(int(n_vertices * 0.2), 2000)  # 20% of vertices or max 2000
        control_indices = np.random.choice(n_vertices, n_control, replace=False)
        control_vertices = source_vertices[control_indices]
        
        print(f"Using {n_control} control points")
        
        # 2. Build deformation graph
        if progress_callback:
            progress_callback("Building deformation graph...", 0.2)
            
        # Build sparse weight matrix using RBF
        tree = cKDTree(control_vertices)
        rows, cols, data = [], [], []
        
        for i in range(n_vertices):
            # Find k nearest control points
            k = min(8, n_control)
            dists, indices = tree.query(source_vertices[i], k=k)
            
            # RBF weights
            sigma = np.median(dists) * 2 if np.median(dists) > 0 else 0.1
            weights = np.exp(-dists**2 / (2 * sigma**2))
            weights = weights / (weights.sum() + 1e-10)
            
            for j, idx in enumerate(indices):
                if weights[j] > 1e-6:
                    rows.append(i)
                    cols.append(idx)
                    data.append(weights[j])
                    
        W = sp.csr_matrix((data, (rows, cols)), shape=(n_vertices, n_control))
        
        # 3. Initialize deformation
        deformation = np.zeros((n_control, 3))
        
        # 4. Multi-resolution optimization
        resolutions = [0.25, 0.5, 1.0] if n_vertices > 10000 else [0.5, 1.0]
        
        for res_idx, resolution in enumerate(resolutions):
            if progress_callback:
                progress_callback(f"Optimizing at resolution {resolution:.2f}...", 
                                0.3 + 0.5 * res_idx / len(resolutions))
            
            # Subsample vertices for this resolution
            n_sample = int(n_vertices * resolution)
            if resolution < 1.0:
                sample_indices = np.random.choice(n_vertices, n_sample, replace=False)
                vertices_subset = source_vertices[sample_indices]
                W_subset = W[sample_indices]
            else:
                vertices_subset = source_vertices
                W_subset = W
                sample_indices = np.arange(n_vertices)
            
            # Build target KDTree
            target_tree = cKDTree(target_mesh.vertices)
            
            # Optimization iterations
            max_iter = int(30 * resolution)
            learning_rate_init = 0.1
            
            for iteration in range(max_iter):
                # Apply current deformation
                deformed_vertices = vertices_subset + W_subset @ deformation
                
                # Find closest points on target mesh
                distances, indices = target_tree.query(deformed_vertices)
                target_correspondences = target_mesh.vertices[indices]
                
                # Compute residuals
                residuals = target_correspondences - deformed_vertices
                
                # Compute gradient
                gradient = W_subset.T @ residuals
                
                # Add regularization (smoothness)
                gradient -= self.lambda_smooth * deformation
                
                # Add detail preservation
                if iteration > 5:  # Let it move first
                    original_displacement = W_subset @ deformation
                    detail_gradient = W_subset.T @ original_displacement
                    gradient -= self.lambda_detail * detail_gradient
                
                # Update with decaying learning rate
                learning_rate = learning_rate_init * np.exp(-iteration * 0.05)
                deformation += learning_rate * gradient
                
                # Print progress
                if iteration % 10 == 0:
                    mean_dist = np.mean(distances)
                    max_dist = np.max(distances)
                    print(f"  Res {resolution:.2f}, Iter {iteration}: "
                          f"mean dist = {mean_dist:.6f}, max = {max_dist:.6f}")
                    
                # Early stopping
                if iteration > 10 and np.mean(distances) < 0.001:
                    print(f"  Converged early at iteration {iteration}")
                    break
        
        # 5. Apply final deformation to all vertices
        if progress_callback:
            progress_callback("Applying final deformation...", 0.9)
            
        final_deformation = W @ deformation
        aligned_vertices = source_vertices + final_deformation
        
        # 6. Create aligned mesh with same topology
        aligned_mesh = trimesh.Trimesh(
            vertices=aligned_vertices,
            faces=source_mesh.faces,
            vertex_normals=source_mesh.vertex_normals if hasattr(source_mesh, 'vertex_normals') else None,
            vertex_colors=source_mesh.visual.vertex_colors if hasattr(source_mesh.visual, 'vertex_colors') else None
        )
        
        # 7. Compute final alignment quality
        final_tree = cKDTree(target_mesh.vertices)
        final_distances, _ = final_tree.query(aligned_vertices)
        
        print("="*50)
        print("Alignment Complete!")
        print(f"Mean distance to target: {np.mean(final_distances):.6f}")
        print(f"Max distance to target: {np.max(final_distances):.6f}")
        print(f"95th percentile: {np.percentile(final_distances, 95):.6f}")
        print("="*50)
        
        if progress_callback:
            progress_callback("Mesh alignment complete!", 1.0)
            
        return aligned_mesh
    
    def _gaussian_kernel(self, X, Y, sigma):
        """Gaussian kernel matrix"""
        pairwise_dists = np.sum(X**2, axis=1)[:, None] + np.sum(Y**2, axis=1) - 2 * X @ Y.T
        return np.exp(-pairwise_dists / (2 * sigma**2))
    
    def automatic_preprocessing(self, points, normals=None, target_size=50000):
        """Automatically preprocess point cloud for optimal alignment"""
        original_size = len(points)
        
        # Downsample if too large
        if len(points) > target_size:
            print(f"Automatically downsampling from {len(points)} to ~{target_size} points")
            
            # Estimate voxel size for target number of points
            bbox_diag = np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0))
            voxel_size = bbox_diag / (target_size ** (1/3)) * 2  # Heuristic
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            if normals is not None and len(normals) > 0:
                pcd.normals = o3d.utility.Vector3dVector(normals)
                
            downsampled = pcd.voxel_down_sample(voxel_size)
            points = np.asarray(downsampled.points)
            
            if downsampled.has_normals():
                normals = np.asarray(downsampled.normals)
            else:
                normals = None
            
            print(f"Downsampled to {len(points)} points (voxel size: {voxel_size:.4f})")
            
        return points, normals
    
    def align_point_clouds(self, source_points, target_points, 
                          source_normals=None, target_normals=None,
                          method='hybrid', progress_callback=None):
        """Main point cloud alignment function"""
        print(f"Aligning point clouds: {len(source_points)} -> {len(target_points)} points")
        
        # Handle large size disparities
        size_ratio = len(source_points) / len(target_points)
        if size_ratio > 100 or size_ratio < 0.01:
            print(f"Warning: Large size disparity (ratio: {size_ratio:.2f})")
            print(f"  Source: {len(source_points):,} points")
            print(f"  Target: {len(target_points):,} points")
            print("Consider using the Downsample button in Preprocessing if performance is poor")
        
        # Auto-estimate correspondence distance if needed
        if self.auto_correspondence:
            self.correspondence_dist = self.estimate_correspondence_distance(source_points, target_points)
        
        # Estimate normals if not provided
        if source_normals is None or len(source_normals) == 0:
            if progress_callback:
                progress_callback("Estimating source normals...", 0.05)
            print("Estimating source normals...")
            source_normals = PointCloudProcessor.estimate_normals(source_points, k_neighbors=min(30, len(source_points)//10))
            
        if target_normals is None or len(target_normals) == 0:
            if progress_callback:
                progress_callback("Estimating target normals...", 0.1)
            print("Estimating target normals...")
            target_normals = PointCloudProcessor.estimate_normals(target_points, k_neighbors=min(30, len(target_points)//10))
        
        # Initial alignment
        if self.use_icp_init and method != 'rigid':
            if progress_callback:
                progress_callback("Computing initial alignment...", 0.2)
            
            try:
                if self.use_fpfh and len(source_points) > 100 and len(target_points) > 100:
                    # Feature-based RANSAC
                    print("Attempting FPFH-based RANSAC alignment...")
                    init_transform = self.initial_alignment_ransac(
                        source_points, target_points, source_normals, target_normals
                    )
                else:
                    print("Skipping FPFH (insufficient points or disabled)")
                    init_transform = np.eye(4)
                    
                # Refine with ICP
                print("Refining with ICP...")
                transform, fitness = self.point_to_point_icp(source_points, target_points, init_transform)
                
                # Apply initial transformation
                source_points_h = np.hstack([source_points, np.ones((len(source_points), 1))])
                source_points = (transform @ source_points_h.T).T[:, :3]
                
                # Rotate normals
                if source_normals is not None:
                    source_normals = (transform[:3, :3] @ source_normals.T).T
                
                print(f"Initial alignment fitness: {fitness:.4f}")
                
            except Exception as e:
                print(f"Initial alignment failed: {e}")
                print("Continuing without initial alignment")
        
        if method == 'rigid':
            # Rigid alignment only
            if progress_callback:
                progress_callback("Running ICP rigid alignment...", 0.5)
            
            # If we haven't done ICP yet, do it now
            if not self.use_icp_init:
                try:
                    print("Running ICP alignment...")
                    transform, fitness = self.point_to_point_icp(source_points, target_points, np.eye(4))
                    
                    # Apply transformation
                    source_points_h = np.hstack([source_points, np.ones((len(source_points), 1))])
                    aligned_points = (transform @ source_points_h.T).T[:, :3]
                    
                    print(f"ICP fitness: {fitness:.4f}")
                except Exception as e:
                    print(f"ICP failed: {e}")
                    aligned_points = source_points
            else:
                # Already transformed
                aligned_points = source_points
            
        elif method == 'cpd':
            # Coherent Point Drift
            if progress_callback:
                progress_callback("Running CPD non-rigid alignment...", 0.4)
            
            # Downsample if too many points for CPD
            max_cpd_points = 10000
            if len(source_points) > max_cpd_points:
                print(f"Downsampling for CPD: {len(source_points)} -> {max_cpd_points}")
                indices = np.random.choice(len(source_points), max_cpd_points, replace=False)
                cpd_source = source_points[indices]
                
                # Run CPD on subset
                cpd_aligned = self.non_rigid_cpd(cpd_source, target_points, w=0.1)
                
                # Interpolate deformation to all points
                from scipy.interpolate import RBFInterpolator
                rbf = RBFInterpolator(cpd_source, cpd_aligned - cpd_source, kernel='thin_plate_spline')
                deformation = rbf(source_points)
                aligned_points = source_points + deformation
            else:
                aligned_points = self.non_rigid_cpd(source_points, target_points, w=0.1)
            
        elif method == 'hybrid':
            # Hybrid approach: rigid + non-rigid refinement
            if progress_callback:
                progress_callback("Building deformation graph...", 0.3)
                
            try:
                # Build adaptive deformation graph
                W, control_indices = self.build_deformation_graph_adaptive(source_points, k_neighbors=min(8, len(source_points)//100))
                n_control = len(control_indices)
                print(f"Using {n_control} control points for deformation")
                
                # Initialize deformation
                deformation = np.zeros((n_control, 3))
                
                # Multi-resolution optimization
                resolutions = [0.25, 0.5, 1.0]
                
                # Adjust resolutions based on point cloud size
                if len(source_points) < 1000:
                    resolutions = [1.0]
                elif len(source_points) < 10000:
                    resolutions = [0.5, 1.0]
                
                for res_idx, resolution in enumerate(resolutions):
                    if progress_callback:
                        progress_callback(f"Optimizing at resolution {resolution:.2f}...", 
                                        0.4 + 0.4 * res_idx / len(resolutions))
                    
                    # Subsample for this resolution
                    n_sample = int(len(source_points) * resolution)
                    n_sample = max(100, min(n_sample, len(source_points)))  # Ensure reasonable bounds
                    
                    if resolution < 1.0:
                        sample_idx = np.random.choice(len(source_points), n_sample, replace=False)
                        points_subset = source_points[sample_idx]
                        normals_subset = source_normals[sample_idx]
                    else:
                        points_subset = source_points
                        normals_subset = source_normals
                        sample_idx = np.arange(len(source_points))
                    
                    # Optimize deformation
                    max_iter_res = max(10, int(self.max_iterations * resolution))
                    for iter in range(max_iter_res):
                        # Apply current deformation
                        if resolution < 1.0:
                            W_subset = W[sample_idx]
                            deformed = points_subset + W_subset @ deformation
                        else:
                            deformed = source_points + W @ deformation
                        
                        # Find correspondences
                        tree = cKDTree(target_points)
                        distances, indices = tree.query(deformed, k=1)
                        
                        # Compute gradients
                        correspondences = target_points[indices]
                        residuals = correspondences - deformed
                        
                        # Update deformation
                        if resolution < 1.0:
                            grad = W_subset.T @ residuals
                        else:
                            grad = W.T @ residuals
                            
                        # Add regularization
                        grad -= self.lambda_smooth * deformation
                        
                        # Update with momentum
                        learning_rate = 0.01 * np.exp(-iter * 0.1)
                        deformation += learning_rate * grad
                        
                        if iter % 10 == 0:
                            mean_dist = np.mean(distances)
                            print(f"    Resolution {resolution:.2f}, Iteration {iter}: mean distance = {mean_dist:.6f}")
                            
                # Apply final deformation
                aligned_points = source_points + W @ deformation
                
            except Exception as e:
                print(f"Hybrid alignment failed: {e}")
                print("Falling back to rigid alignment result")
                aligned_points = source_points
            
        else:
            raise ValueError(f"Unknown alignment method: {method}")
            
        if progress_callback:
            progress_callback("Alignment complete!", 1.0)
            
        return aligned_points
    
    def align_hybrid(self, source_data, target_data, source_type, target_type, 
                    progress_callback=None):
        """Align any combination of meshes and point clouds"""
        
        print(f"Aligning {source_type} to {target_type}")
        
        # Handle mesh-to-mesh alignment with dedicated algorithm
        if source_type == 'mesh' and target_type == 'mesh':
            print(f"Mesh-to-mesh alignment: {len(source_data.vertices)} -> {len(target_data.vertices)} vertices")
            return self.align_mesh_to_mesh(source_data, target_data, progress_callback)
        
        # Convert to point clouds for mixed or point-to-point alignment
        if source_type == 'mesh':
            # Sample points from mesh surface
            n_samples = min(len(source_data.vertices) * 5, 100000)
            source_points = source_data.sample(n_samples)
            source_normals = None  # Will be estimated
            print(f"Sampled {len(source_points)} points from source mesh")
        else:
            source_points = source_data['points']
            source_normals = source_data.get('normals', None)
            
        if target_type == 'mesh':
            # For mesh targets, use vertices directly
            target_points = target_data.vertices
            target_normals = target_data.vertex_normals if hasattr(target_data, 'vertex_normals') else None
            print(f"Using {len(target_points)} vertices from target mesh")
        else:
            target_points = target_data['points']
            target_normals = target_data.get('normals', None)
            
        # Run alignment based on point cloud conversion
        aligned_points = self.align_point_clouds(
            source_points, target_points, 
            source_normals, target_normals,
            method='hybrid',  # Use hybrid method for mixed types
            progress_callback=progress_callback
        )
        
        # Return appropriate format
        if source_type == 'mesh':
            # Deform original mesh vertices based on point cloud alignment
            # This preserves the mesh topology
            tree = cKDTree(source_points)
            distances, indices = tree.query(source_data.vertices, k=5)
            
            # Weighted average of deformations
            deformations = aligned_points - source_points
            vertex_deformations = []
            
            for i in range(len(source_data.vertices)):
                weights = 1.0 / (distances[i] + 1e-6)
                weights = weights / weights.sum()
                deform = np.sum(deformations[indices[i]] * weights[:, None], axis=0)
                vertex_deformations.append(deform)
                
            vertex_deformations = np.array(vertex_deformations)
            aligned_vertices = source_data.vertices + vertex_deformations
            
            return trimesh.Trimesh(
                vertices=aligned_vertices,
                faces=source_data.faces,
                vertex_normals=source_data.vertex_normals if hasattr(source_data, 'vertex_normals') else None,
                vertex_colors=source_data.visual.vertex_colors if hasattr(source_data.visual, 'vertex_colors') else None
            )
        else:
            return {'points': aligned_points, 'normals': source_normals}
