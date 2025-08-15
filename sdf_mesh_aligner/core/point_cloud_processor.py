#!/usr/bin/env python3
"""
Point cloud processing utilities for SDF Mesh Aligner
"""
import os
import numpy as np
import trimesh
import open3d as o3d

class PointCloudProcessor:
    """Utilities for point cloud processing"""
    
    @staticmethod
    def load_point_cloud(filename):
        """Load point cloud from various formats"""
        ext = os.path.splitext(filename)[1].lower()
        
        if ext in ['.ply', '.pcd', '.xyz', '.pts']:
            # Use Open3D for point cloud formats
            pcd = o3d.io.read_point_cloud(filename)
            points = np.asarray(pcd.points)
            normals = np.asarray(pcd.normals) if pcd.has_normals() else None
            colors = np.asarray(pcd.colors) if pcd.has_colors() else None
            return points, normals, colors
            
        elif ext in ['.txt', '.csv']:
            # Load ASCII point clouds
            data = np.loadtxt(filename)
            if data.shape[1] >= 3:
                points = data[:, :3]
                normals = data[:, 3:6] if data.shape[1] >= 6 else None
                colors = data[:, 6:9] if data.shape[1] >= 9 else None
                return points, normals, colors
            else:
                raise ValueError("File must have at least 3 columns (x, y, z)")
                
        elif ext in ['.las', '.laz']:
            # LAS/LAZ format (requires laspy)
            try:
                import laspy
                las = laspy.read(filename)
                points = np.vstack([las.x, las.y, las.z]).T
                normals = None
                colors = None
                if hasattr(las, 'red'):
                    colors = np.vstack([las.red, las.green, las.blue]).T / 65535.0
                return points, normals, colors
            except ImportError:
                raise ImportError("Please install laspy to read LAS/LAZ files: pip install laspy")
                
        else:
            raise ValueError(f"Unsupported point cloud format: {ext}")
    
    @staticmethod
    def estimate_normals(points, k_neighbors=30):
        """Estimate normals for point cloud using PCA"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(k_neighbors))
        pcd.orient_normals_consistent_tangent_plane(k_neighbors)
        return np.asarray(pcd.normals)
    
    @staticmethod
    def voxel_downsample(points, voxel_size, normals=None, colors=None):
        """Downsample point cloud using voxel grid"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
        downsampled = pcd.voxel_down_sample(voxel_size)
        
        result = [np.asarray(downsampled.points)]
        if normals is not None:
            result.append(np.asarray(downsampled.normals))
        if colors is not None:
            result.append(np.asarray(downsampled.colors))
            
        return tuple(result) if len(result) > 1 else result[0]
    
    @staticmethod
    def statistical_outlier_removal(points, nb_neighbors=20, std_ratio=2.0):
        """Remove outliers using statistical analysis"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        cleaned, ind = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
        return np.asarray(cleaned.points), ind
    
    @staticmethod
    def points_to_mesh(points, method='poisson', depth=9, normals=None):
        """Convert point cloud to mesh using various methods"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if normals is None:
            pcd.estimate_normals()
        else:
            pcd.normals = o3d.utility.Vector3dVector(normals)
            
        if method == 'poisson':
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=depth, width=0, scale=1.1, linear_fit=False
            )
        elif method == 'ball_pivoting':
            radii = [0.005, 0.01, 0.02, 0.04]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii)
            )
        elif method == 'alpha':
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 0.03)
        else:
            raise ValueError(f"Unknown meshing method: {method}")
            
        # Convert to trimesh
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        return trimesh.Trimesh(vertices=vertices, faces=faces)
