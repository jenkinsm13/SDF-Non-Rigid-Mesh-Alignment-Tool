#!/usr/bin/env python3
"""
Hybrid GUI for mesh and point cloud alignment
"""
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import time
import os
import numpy as np
import trimesh
import open3d as o3d

from ..core.hybrid_aligner import HybridAligner
from ..core.point_cloud_processor import PointCloudProcessor
from ..core.mesh_utils import MeshUtils
from ..config.settings import ConfigManager
from ..utils.performance import PerformanceMonitor
from ..utils.validation import validate_mesh, validate_config

class HybridAlignmentGUI:
    """Simple, functional GUI for mesh and point cloud alignment"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Mesh & Point Cloud Alignment Tool")
        self.root.geometry("1500x900")
        
        # Initialize variables
        self.source_data = None
        self.target_data = None
        self.aligned_data = None
        self.source_type = None  # 'mesh' or 'pointcloud'
        self.target_type = None
        self.aligner = HybridAligner()
        
        # Setup GUI
        self.setup_menu()
        self.setup_widgets()
        
    def setup_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        
        source_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Load Source", menu=source_menu)
        source_menu.add_command(label="Load Mesh", command=lambda: self.load_source('mesh'))
        source_menu.add_command(label="Load Point Cloud", command=lambda: self.load_source('pointcloud'))
        
        target_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Load Target", menu=target_menu)
        target_menu.add_command(label="Load Mesh", command=lambda: self.load_target('mesh'))
        target_menu.add_command(label="Load Point Cloud", command=lambda: self.load_target('pointcloud'))
        
        file_menu.add_separator()
        file_menu.add_command(label="Save Aligned Data", command=self.save_aligned)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Convert Points to Mesh", command=self.convert_to_mesh)
        tools_menu.add_command(label="Downsample Point Cloud", command=self.downsample_dialog)
        tools_menu.add_command(label="Remove Outliers", command=self.remove_outliers)
        tools_menu.add_command(label="Estimate Normals", command=self.estimate_normals)
        
        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="Algorithm Parameters", command=self.show_settings)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        
    def setup_widgets(self):
        """Create main widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Data loading section
        load_frame = ttk.LabelFrame(control_frame, text="Data Loading", padding="5")
        load_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Source controls
        ttk.Label(load_frame, text="Source:").grid(row=0, column=0, sticky=tk.W)
        source_btn_frame = ttk.Frame(load_frame)
        source_btn_frame.grid(row=0, column=1, padx=5)
        ttk.Button(source_btn_frame, text="Load Mesh", 
                  command=lambda: self.load_source('mesh')).pack(side=tk.LEFT, padx=2)
        ttk.Button(source_btn_frame, text="Load Point Cloud", 
                  command=lambda: self.load_source('pointcloud')).pack(side=tk.LEFT, padx=2)
        
        self.source_label = ttk.Label(load_frame, text="No source loaded", foreground="red")
        self.source_label.grid(row=0, column=2, padx=10)
        self.source_info = ttk.Label(load_frame, text="", font=("Courier", 9))
        self.source_info.grid(row=0, column=3, padx=5)
        
        # Target controls
        ttk.Label(load_frame, text="Target:").grid(row=1, column=0, sticky=tk.W)
        target_btn_frame = ttk.Frame(load_frame)
        target_btn_frame.grid(row=1, column=1, padx=5)
        ttk.Button(target_btn_frame, text="Load Mesh", 
                  command=lambda: self.load_target('mesh')).pack(side=tk.LEFT, padx=2)
        ttk.Button(target_btn_frame, text="Load Point Cloud", 
                  command=lambda: self.load_target('pointcloud')).pack(side=tk.LEFT, padx=2)
        
        self.target_label = ttk.Label(load_frame, text="No target loaded", foreground="red")
        self.target_label.grid(row=1, column=2, padx=10)
        self.target_info = ttk.Label(load_frame, text="", font=("Courier", 9))
        self.target_info.grid(row=1, column=3, padx=5)
        
        # Alignment method selection
        method_frame = ttk.LabelFrame(control_frame, text="Alignment Method", padding="5")
        method_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.method_var = tk.StringVar(value="hybrid")
        
        # Add description
        ttk.Label(method_frame, text="Choose alignment type:", font=("Arial", 9)).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0,5))
        
        ttk.Radiobutton(method_frame, text="Rigid (rotation/translation only)", variable=self.method_var, 
                       value="rigid").grid(row=1, column=0, sticky=tk.W, padx=5)
        ttk.Radiobutton(method_frame, text="Non-rigid (deformable)", variable=self.method_var, 
                       value="cpd").grid(row=1, column=1, sticky=tk.W, padx=5)
        ttk.Radiobutton(method_frame, text="Hybrid (best for soft bodies)", variable=self.method_var, 
                       value="hybrid").grid(row=1, column=2, sticky=tk.W, padx=5)
        
        # Algorithm parameters
        params_frame = ttk.LabelFrame(control_frame, text="Algorithm Parameters", padding="5")
        params_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Smoothness
        ttk.Label(params_frame, text="Smoothness:").grid(row=0, column=0, sticky=tk.W)
        self.lambda_smooth_var = tk.DoubleVar(value=0.01)  # Lower default for stronger deformation
        scale = ttk.Scale(params_frame, from_=0.0, to=0.5, variable=self.lambda_smooth_var,
                 orient=tk.HORIZONTAL, length=150)
        scale.grid(row=0, column=1)
        label = ttk.Label(params_frame, text="0.010")
        label.grid(row=0, column=2)
        ttk.Label(params_frame, text="(lower = finer details)", 
                 font=("Arial", 8, "italic")).grid(row=0, column=3, sticky=tk.W, padx=(5,0))
        
        # Update label when scale changes
        def update_smooth_label(val):
            label.config(text=f"{float(val):.3f}")
        scale.config(command=update_smooth_label)
        
        # Stiffness (inverse of deformation strength)
        ttk.Label(params_frame, text="Stiffness:").grid(row=1, column=0, sticky=tk.W)
        self.lambda_detail_var = tk.DoubleVar(value=0.1)  # Lower = more deformation allowed
        scale2 = ttk.Scale(params_frame, from_=0.0, to=1.0, variable=self.lambda_detail_var,
                 orient=tk.HORIZONTAL, length=150)
        scale2.grid(row=1, column=1)
        label2 = ttk.Label(params_frame, text="0.100")
        label2.grid(row=1, column=2)
        ttk.Label(params_frame, text="(lower = stronger deformation)", 
                 font=("Arial", 8, "italic")).grid(row=1, column=3, sticky=tk.W, padx=(5,0))
        
        # Update label when scale changes
        def update_stiff_label(val):
            label2.config(text=f"{float(val):.3f}")
        scale2.config(command=update_stiff_label)
        
        # Correspondence distance
        ttk.Label(params_frame, text="Max Correspondence:").grid(row=2, column=0, sticky=tk.W)
        self.corr_dist_var = tk.DoubleVar(value=0.1)
        scale3 = ttk.Scale(params_frame, from_=0.01, to=1.0, variable=self.corr_dist_var,
                 orient=tk.HORIZONTAL, length=150)
        scale3.grid(row=2, column=1)
        label3 = ttk.Label(params_frame, text="0.100")
        label3.grid(row=2, column=2)
        
        def update_corr_label(val):
            label3.config(text=f"{float(val):.3f}")
        scale3.config(command=update_corr_label)
        
        # Iterations
        ttk.Label(params_frame, text="Max Iterations:").grid(row=3, column=0, sticky=tk.W)
        self.max_iter_var = tk.IntVar(value=100)  # Increased default
        ttk.Spinbox(params_frame, from_=50, to=500, increment=50, textvariable=self.max_iter_var,
                   width=10).grid(row=3, column=1, sticky=tk.W)
        
        # Use features
        self.use_features_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Use FPFH Features", 
                       variable=self.use_features_var).grid(row=4, column=0, columnspan=2, sticky=tk.W)
        
        # Point cloud preprocessing
        preprocess_frame = ttk.LabelFrame(control_frame, text="Preprocessing (Point Clouds Only)", padding="5")
        preprocess_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.downsample_btn = ttk.Button(preprocess_frame, text="Downsample", 
                                        command=self.downsample_dialog, state=tk.DISABLED)
        self.downsample_btn.grid(row=0, column=0, padx=2)
        
        self.outliers_btn = ttk.Button(preprocess_frame, text="Remove Outliers", 
                                       command=self.remove_outliers, state=tk.DISABLED)
        self.outliers_btn.grid(row=0, column=1, padx=2)
        
        self.normals_btn = ttk.Button(preprocess_frame, text="Estimate Normals", 
                                      command=self.estimate_normals, state=tk.DISABLED)
        self.normals_btn.grid(row=0, column=2, padx=2)
        
        self.convert_btn = ttk.Button(preprocess_frame, text="Convert to Mesh", 
                                      command=self.convert_to_mesh, state=tk.DISABLED)
        self.convert_btn.grid(row=0, column=3, padx=2)
        
        # Align button
        self.align_button = ttk.Button(control_frame, text="ALIGN", 
                                      command=self.run_alignment, state=tk.DISABLED,
                                      style="Accent.TButton")
        self.align_button.grid(row=4, column=0, columnspan=2, pady=20)
        
        # Progress
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var,
                                           length=300, mode='determinate')
        self.progress_bar.grid(row=5, column=0, columnspan=2, pady=5)
        
        self.progress_label = ttk.Label(control_frame, text="")
        self.progress_label.grid(row=6, column=0, columnspan=2)
        
        # Save button
        self.save_button = ttk.Button(control_frame, text="Save Aligned Data",
                                     command=self.save_aligned, state=tk.DISABLED)
        self.save_button.grid(row=7, column=0, columnspan=2, pady=10)
        
        # Visualization
        viz_frame = ttk.LabelFrame(main_frame, text="3D Visualization", padding="10")
        viz_frame.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Viz controls
        viz_controls = ttk.Frame(viz_frame)
        viz_controls.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        ttk.Label(viz_controls, text="Display:").pack(side=tk.LEFT, padx=5)
        self.display_mode = tk.StringVar(value="all")
        ttk.Radiobutton(viz_controls, text="All", variable=self.display_mode,
                       value="all", command=self.update_visualization).pack(side=tk.LEFT)
        ttk.Radiobutton(viz_controls, text="Source", variable=self.display_mode,
                       value="source", command=self.update_visualization).pack(side=tk.LEFT)
        ttk.Radiobutton(viz_controls, text="Target", variable=self.display_mode,
                       value="target", command=self.update_visualization).pack(side=tk.LEFT)
        ttk.Radiobutton(viz_controls, text="Aligned", variable=self.display_mode,
                       value="aligned", command=self.update_visualization).pack(side=tk.LEFT)
        ttk.Radiobutton(viz_controls, text="Overlay", variable=self.display_mode,
                       value="overlay", command=self.update_visualization).pack(side=tk.LEFT)
        
        # Point size control
        ttk.Label(viz_controls, text="Point Size:").pack(side=tk.LEFT, padx=(20, 5))
        self.point_size_var = tk.IntVar(value=1)
        ttk.Scale(viz_controls, from_=0.1, to=5, variable=self.point_size_var,
                 orient=tk.HORIZONTAL, length=100, command=lambda x: self.update_visualization()).pack(side=tk.LEFT)
        
        # Matplotlib figure
        self.fig = plt.figure(figsize=(10, 7))
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Statistics panel
        stats_frame = ttk.LabelFrame(main_frame, text="Statistics & Log", padding="10")
        stats_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Stats display
        self.stats_text = tk.Text(stats_frame, height=8, width=50, wrap=tk.WORD)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        
        # Initial stats and button states
        self.update_stats()
        self.update_preprocessing_buttons()
        
        # Show quick start guide
        self.show_quick_start()
        

        
    def load_source(self, data_type):
        """Load source data (mesh or point cloud)"""
        if data_type == 'mesh':
            filetypes = [("Mesh files", "*.ply *.obj *.stl *.off *.glb"), ("All files", "*.*")]
            title = "Select Source Mesh"
        else:
            filetypes = [("Point clouds", "*.ply *.pcd *.xyz *.pts *.txt *.csv *.las *.laz"), 
                        ("All files", "*.*")]
            title = "Select Source Point Cloud"
            
        filename = filedialog.askopenfilename(title=title, filetypes=filetypes)
        
        if filename:
            try:
                if data_type == 'mesh':
                    # Load as mesh
                    self.source_data = trimesh.load(filename, force='mesh')
                    if isinstance(self.source_data, trimesh.Scene):
                        meshes = list(self.source_data.geometry.values())
                        self.source_data = trimesh.util.concatenate(meshes)
                    
                    # Ensure it's a valid mesh
                    if not hasattr(self.source_data, 'vertices') or not hasattr(self.source_data, 'faces'):
                        raise ValueError("Invalid mesh file")
                        
                    self.source_type = 'mesh'
                    info = f"{len(self.source_data.vertices):,} verts, {len(self.source_data.faces):,} faces"
                    
                    print(f"Loaded source MESH: {os.path.basename(filename)}")
                    print(f"  Vertices: {len(self.source_data.vertices):,}")
                    print(f"  Faces: {len(self.source_data.faces):,}")
                    
                else:
                    # Load as point cloud
                    points, normals, colors = PointCloudProcessor.load_point_cloud(filename)
                    self.source_data = {'points': points, 'normals': normals, 'colors': colors}
                    self.source_type = 'pointcloud'
                    info = f"{len(points):,} points"
                    
                    print(f"Loaded source POINT CLOUD: {os.path.basename(filename)}")
                    print(f"  Points: {len(points):,}")
                    
                self.source_label.config(text=f"{data_type.capitalize()}: {os.path.basename(filename)}", 
                                       foreground="green")
                self.source_info.config(text=info)
                
                self.update_stats()
                self.update_visualization()
                self.check_ready()
                self.update_preprocessing_buttons()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load source: {str(e)}")
                
    def load_target(self, data_type):
        """Load target data (mesh or point cloud)"""
        if data_type == 'mesh':
            filetypes = [("Mesh files", "*.ply *.obj *.stl *.off *.glb"), ("All files", "*.*")]
            title = "Select Target Mesh"
        else:
            filetypes = [("Point clouds", "*.ply *.pcd *.xyz *.pts *.txt *.csv *.las *.laz"), 
                        ("All files", "*.*")]
            title = "Select Target Point Cloud"
            
        filename = filedialog.askopenfilename(title=title, filetypes=filetypes)
        
        if filename:
            try:
                if data_type == 'mesh':
                    # Load as mesh
                    self.target_data = trimesh.load(filename, force='mesh')
                    if isinstance(self.target_data, trimesh.Scene):
                        meshes = list(self.target_data.geometry.values())
                        self.target_data = trimesh.util.concatenate(meshes)
                    
                    # Ensure it's a valid mesh
                    if not hasattr(self.target_data, 'vertices') or not hasattr(self.target_data, 'faces'):
                        raise ValueError("Invalid mesh file")
                        
                    self.target_type = 'mesh'
                    info = f"{len(self.target_data.vertices):,} verts, {len(self.target_data.faces):,} faces"
                    
                    print(f"Loaded target MESH: {os.path.basename(filename)}")
                    print(f"  Vertices: {len(self.target_data.vertices):,}")
                    print(f"  Faces: {len(self.target_data.faces):,}")
                    
                else:
                    # Load as point cloud
                    points, normals, colors = PointCloudProcessor.load_point_cloud(filename)
                    self.target_data = {'points': points, 'normals': normals, 'colors': colors}
                    self.target_type = 'pointcloud'
                    info = f"{len(points):,} points"
                    
                    print(f"Loaded target POINT CLOUD: {os.path.basename(filename)}")
                    print(f"  Points: {len(points):,}")
                    
                self.target_label.config(text=f"{data_type.capitalize()}: {os.path.basename(filename)}", 
                                       foreground="green")
                self.target_info.config(text=info)
                
                self.update_stats()
                self.update_visualization()
                self.check_ready()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load target: {str(e)}")
                
    def update_preprocessing_buttons(self):
        """Enable/disable preprocessing buttons based on data type"""
        if self.source_type == 'pointcloud':
            self.downsample_btn.config(state=tk.NORMAL)
            self.outliers_btn.config(state=tk.NORMAL)
            self.normals_btn.config(state=tk.NORMAL)
            self.convert_btn.config(state=tk.NORMAL)
        else:
            self.downsample_btn.config(state=tk.DISABLED)
            self.outliers_btn.config(state=tk.DISABLED)
            self.normals_btn.config(state=tk.DISABLED)
            self.convert_btn.config(state=tk.DISABLED)
    
    def check_ready(self):
        """Check if ready to align"""
        if self.source_data is not None and self.target_data is not None:
            self.align_button.config(state=tk.NORMAL)
            
            # Update method options based on data types
            if self.source_type == 'mesh' and self.target_type == 'mesh':
                # For mesh-to-mesh, hybrid is best
                self.method_var.set('hybrid')
                print("\nMesh-to-mesh alignment ready")
                print("Recommended method: Hybrid (non-rigid deformation)")
            elif self.source_type == 'pointcloud' and self.target_type == 'pointcloud':
                print("\nPoint cloud alignment ready")
            else:
                print("\nMixed data types - will use adaptive alignment")
        else:
            self.align_button.config(state=tk.DISABLED)
        
        # Always update preprocessing buttons based on current source
        if hasattr(self, 'downsample_btn'):
            self.update_preprocessing_buttons()
            
    def show_quick_start(self):
        """Show quick start guide in stats window"""
        self.stats_text.delete(1.0, tk.END)
        
        quick_start = """QUICK START GUIDE
=====================================

1. LOAD DATA:
   • Click "Load Mesh" or "Load Point Cloud" 
     for both Source and Target
   • Supported: PLY, PCD, OBJ, STL, XYZ, LAS

2. PREPROCESS (if needed):
   • Downsample: Reduce point density
   • Remove Outliers: Clean noisy data
   • Estimate Normals: Required for FPFH

3. CHOOSE METHOD:
   • Rigid: Fast ICP alignment only
   • Non-rigid: CPD for deformable objects
   • Hybrid: Best for soft-body alignment

4. ADJUST PARAMETERS:
   • Smoothness: Higher = smoother deformation
   • Max Correspondence: Search radius

5. CLICK "ALIGN" TO START

TIPS:
• For large size differences, downsample first
• Disable FPFH if alignment fails
• Try Rigid method for initial testing
• Check console for detailed progress

=====================================
"""
        self.stats_text.insert(tk.END, quick_start)
    
    def update_stats(self):
        """Update statistics display"""
        self.stats_text.delete(1.0, tk.END)
        
        self.stats_text.insert(tk.END, "DATA STATISTICS\n" + "="*40 + "\n\n")
        
        if self.source_data is not None:
            self.stats_text.insert(tk.END, f"SOURCE ({self.source_type}):\n")
            if self.source_type == 'mesh':
                self.stats_text.insert(tk.END, f"  Vertices: {len(self.source_data.vertices):,}\n")
                self.stats_text.insert(tk.END, f"  Faces: {len(self.source_data.faces):,}\n")
                bounds = self.source_data.bounds
                self.stats_text.insert(tk.END, f"  Bounding box: {bounds[1] - bounds[0]}\n")
            else:
                points = self.source_data['points']
                self.stats_text.insert(tk.END, f"  Points: {len(points):,}\n")
                self.stats_text.insert(tk.END, f"  Has normals: {self.source_data['normals'] is not None}\n")
                self.stats_text.insert(tk.END, f"  Has colors: {self.source_data['colors'] is not None}\n")
                bbox = np.max(points, axis=0) - np.min(points, axis=0)
                self.stats_text.insert(tk.END, f"  Bounding box: [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}]\n")
                
        if self.target_data is not None:
            self.stats_text.insert(tk.END, f"\nTARGET ({self.target_type}):\n")
            if self.target_type == 'mesh':
                self.stats_text.insert(tk.END, f"  Vertices: {len(self.target_data.vertices):,}\n")
                self.stats_text.insert(tk.END, f"  Faces: {len(self.target_data.faces):,}\n")
                bounds = self.target_data.bounds
                self.stats_text.insert(tk.END, f"  Bounding box: {bounds[1] - bounds[0]}\n")
            else:
                points = self.target_data['points']
                self.stats_text.insert(tk.END, f"  Points: {len(points):,}\n")
                self.stats_text.insert(tk.END, f"  Has normals: {self.target_data['normals'] is not None}\n")
                self.stats_text.insert(tk.END, f"  Has colors: {self.target_data['colors'] is not None}\n")
                bbox = np.max(points, axis=0) - np.min(points, axis=0)
                self.stats_text.insert(tk.END, f"  Bounding box: [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}]\n")
                
        if self.source_data is None and self.target_data is None:
            self.show_quick_start()
                
    def update_visualization(self):
        """Update 3D visualization"""
        self.fig.clear()
        
        mode = self.display_mode.get()
        point_size = self.point_size_var.get()
        
        if mode == "all":
            n_plots = sum([self.source_data is not None,
                          self.target_data is not None,
                          self.aligned_data is not None])
            
            if n_plots == 0:
                return
                
            plot_idx = 1
            
            if self.source_data is not None:
                ax = self.fig.add_subplot(1, n_plots, plot_idx, projection='3d')
                self.plot_data(ax, self.source_data, self.source_type, 'blue', 
                             "Source", point_size)
                plot_idx += 1
                
            if self.target_data is not None:
                ax = self.fig.add_subplot(1, n_plots, plot_idx, projection='3d')
                self.plot_data(ax, self.target_data, self.target_type, 'red', 
                             "Target", point_size)
                plot_idx += 1
                
            if self.aligned_data is not None:
                ax = self.fig.add_subplot(1, n_plots, plot_idx, projection='3d')
                data_type = self.source_type  # Aligned has same type as source
                self.plot_data(ax, self.aligned_data, data_type, 'green', 
                             "Aligned", point_size)
                
        elif mode == "overlay" and self.aligned_data is not None:
            ax = self.fig.add_subplot(111, projection='3d')
            self.plot_data(ax, self.target_data, self.target_type, 'red', 
                         "Target", point_size, alpha=0.3)
            data_type = self.source_type
            self.plot_data(ax, self.aligned_data, data_type, 'green', 
                         "Aligned", point_size, alpha=0.7)
            ax.set_title("Overlay: Target (red) vs Aligned (green)")
            
        else:
            ax = self.fig.add_subplot(111, projection='3d')
            
            if mode == "source" and self.source_data is not None:
                self.plot_data(ax, self.source_data, self.source_type, 'blue', 
                             "Source", point_size)
            elif mode == "target" and self.target_data is not None:
                self.plot_data(ax, self.target_data, self.target_type, 'red', 
                             "Target", point_size)
            elif mode == "aligned" and self.aligned_data is not None:
                data_type = self.source_type
                self.plot_data(ax, self.aligned_data, data_type, 'green', 
                             "Aligned", point_size)
                
        self.fig.tight_layout()
        self.canvas.draw()
        
    def plot_data(self, ax, data, data_type, color, title, point_size=1, alpha=0.5):
        """Plot mesh or point cloud data"""
        if data_type == 'mesh':
            vertices = data.vertices
            
            # Plot mesh wireframe for small meshes, points for large ones
            if len(data.faces) < 5000:
                # Plot wireframe
                ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                               triangles=data.faces, 
                               color=color, alpha=alpha*0.7,
                               edgecolor='gray', linewidth=0.1)
                ax.set_title(f"{title} (Mesh)")
            else:
                # Plot vertices as points for large meshes
                # Subsample for visualization if needed
                if len(vertices) > 10000:
                    indices = np.random.choice(len(vertices), 10000, replace=False)
                    vertices = vertices[indices]
                
                ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                          c=color, s=point_size, alpha=alpha, marker='.')
                ax.set_title(f"{title} (Mesh - {len(data.vertices):,} vertices)")
        else:
            # Point cloud
            vertices = data['points']
            
            # Use colors if available
            if data.get('colors') is not None and color == 'blue':
                color = data['colors']
                if len(vertices) > 10000:
                    indices = np.random.choice(len(vertices), 10000, replace=False)
                    vertices = vertices[indices]
                    color = color[indices] if len(color) == len(data['points']) else color
            elif len(vertices) > 10000:
                indices = np.random.choice(len(vertices), 10000, replace=False)
                vertices = vertices[indices]
                
            ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                      c=color, s=point_size, alpha=alpha, marker='.')
            ax.set_title(f"{title} (Points - {len(data['points']):,})")
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Equal aspect ratio
        max_range = np.array([vertices[:, 0].max()-vertices[:, 0].min(),
                             vertices[:, 1].max()-vertices[:, 1].min(),
                             vertices[:, 2].max()-vertices[:, 2].min()]).max() / 2.0
        
        mid_x = (vertices[:, 0].max()+vertices[:, 0].min()) * 0.5
        mid_y = (vertices[:, 1].max()+vertices[:, 1].min()) * 0.5
        mid_z = (vertices[:, 2].max()+vertices[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
    def progress_callback(self, message, progress):
        """Update progress"""
        self.progress_var.set(progress * 100)
        self.progress_label.config(text=message)
        self.root.update_idletasks()
        
    def run_alignment(self):
        """Run alignment in thread"""
        # Check for potential issues before starting
        if self.source_type == 'mesh' and self.target_type == 'mesh':
            source_size = len(self.source_data.vertices)
            target_size = len(self.target_data.vertices)
            size_ratio = max(source_size, target_size) / min(source_size, target_size)
            
            print(f"\nPreparing mesh-to-mesh alignment:")
            print(f"  Source: {source_size:,} vertices")
            print(f"  Target: {target_size:,} vertices")
            print(f"  Size ratio: {size_ratio:.2f}x")
            
            # Only warn if there's a significant size difference
            if size_ratio > 10:
                response = messagebox.askyesno(
                    "Size Difference Detected",
                    f"The meshes have different resolutions:\n"
                    f"Source: {source_size:,} vertices\n"
                    f"Target: {target_size:,} vertices\n"
                    f"Ratio: {size_ratio:.2f}x\n\n"
                    f"The alignment will still work but may take longer.\n"
                    f"Continue?",
                    icon='info'
                )
                if not response:
                    return
                    
        elif self.source_type == 'pointcloud' and self.target_type == 'pointcloud':
            source_size = len(self.source_data['points'])
            target_size = len(self.target_data['points'])
            size_ratio = max(source_size, target_size) / min(source_size, target_size)
            
            if size_ratio > 10:
                response = messagebox.askyesno(
                    "Size Difference Detected",
                    f"Point cloud size difference:\n"
                    f"Source: {source_size:,} points\n"
                    f"Target: {target_size:,} points\n"
                    f"Ratio: {size_ratio:.2f}x\n\n"
                    f"You may want to use the Downsample button\n"
                    f"in Preprocessing to improve performance.\n\n"
                    f"Continue without downsampling?",
                    icon='info'
                )
                if not response:
                    return
        
        # Update parameters - now includes ALL parameters
        self.aligner.lambda_smooth = self.lambda_smooth_var.get()
        self.aligner.lambda_detail = self.lambda_detail_var.get()
        self.aligner.max_iterations = self.max_iter_var.get()
        self.aligner.correspondence_dist = self.corr_dist_var.get()
        self.aligner.use_fpfh = self.use_features_var.get()
        
        print(f"\nAlignment parameters:")
        print(f"  Smoothness: {self.aligner.lambda_smooth:.3f}")
        print(f"  Stiffness: {self.aligner.lambda_detail:.3f}")
        print(f"  Max iterations: {self.aligner.max_iterations}")
        print(f"  Correspondence dist: {self.aligner.correspondence_dist:.3f}")
        
        # Disable controls
        self.align_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        
        # Clear any previous error
        if hasattr(self, 'alignment_error'):
            delattr(self, 'alignment_error')
        
        # Run in thread
        thread = threading.Thread(target=self.alignment_worker)
        thread.daemon = True  # Make thread daemon so it closes with main window
        thread.start()
        
    def alignment_worker(self):
        """Worker thread for alignment"""
        try:
            start_time = time.time()
            
            # Run alignment based on data types
            method = self.method_var.get()
            
            print(f"\nAlignment Configuration:")
            print(f"  Source type: {self.source_type}")
            print(f"  Target type: {self.target_type}")
            print(f"  Method: {method}")
            
            # Route to appropriate alignment method
            if self.source_type == 'mesh' and self.target_type == 'mesh':
                # MESH-TO-MESH alignment
                print("\nUsing dedicated mesh-to-mesh alignment...")
                
                # Don't override user settings - they're already set in run_alignment()
                
                # For mesh-to-mesh, always use the hybrid aligner
                self.aligned_data = self.aligner.align_hybrid(
                    self.source_data, self.target_data,
                    'mesh', 'mesh',
                    progress_callback=self.progress_callback
                )
                
            elif self.source_type == 'pointcloud' and self.target_type == 'pointcloud':
                # POINT-TO-POINT alignment
                source_points = self.source_data['points']
                target_points = self.target_data['points']
                source_normals = self.source_data.get('normals')
                target_normals = self.target_data.get('normals')
                
                # Just warn about size disparity, don't auto-downsample
                size_ratio = len(source_points) / len(target_points)
                if size_ratio > 10:
                    print(f"\nSize disparity detected:")
                    print(f"  Source: {len(source_points):,} points")
                    print(f"  Target: {len(target_points):,} points")  
                    print(f"  Ratio: {size_ratio:.2f}x")
                    print("Consider using the Downsample button if performance is poor")
                
                aligned_points = self.aligner.align_point_clouds(
                    source_points,
                    target_points,
                    source_normals,
                    target_normals,
                    method=method,
                    progress_callback=self.progress_callback
                )
                
                self.aligned_data = {
                    'points': aligned_points,
                    'normals': source_normals,
                    'colors': self.source_data.get('colors')
                }
                
            else:
                # HYBRID alignment (mesh/point combinations)
                print("\nUsing hybrid alignment for mixed data types...")
                self.aligned_data = self.aligner.align_hybrid(
                    self.source_data, self.target_data,
                    self.source_type, self.target_type,
                    progress_callback=self.progress_callback
                )
                
            elapsed = time.time() - start_time
            
            # Compute metrics
            if self.source_type == 'mesh':
                n_points = len(self.aligned_data.vertices)
            else:
                n_points = len(self.aligned_data['points'])
                
            print(f"\n{'='*50}")
            print(f"ALIGNMENT COMPLETE")
            print(f"Time: {elapsed:.2f} seconds")
            print(f"Points/vertices processed: {n_points:,}")
            print(f"Method: {method}")
            print(f"{'='*50}")
            
            self.root.after(0, self.alignment_complete)
            
        except Exception as e:
            error_msg = f"Alignment failed: {str(e)}"
            print(f"\n{'='*50}")
            print("ALIGNMENT FAILED")
            print(error_msg)
            print(f"{'='*50}")
            
            import traceback
            traceback.print_exc()
            
            # Store error for display
            self.alignment_error = error_msg
            self.root.after(0, self.alignment_failed)
            
    def alignment_complete(self):
        """Called when alignment completes"""
        self.align_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.NORMAL)
        self.update_visualization()
        messagebox.showinfo("Success", "Alignment completed successfully!")
        
    def alignment_failed(self):
        """Called when alignment fails"""
        self.align_button.config(state=tk.NORMAL)
        self.progress_var.set(0)
        self.progress_label.config(text="Alignment failed")
        
        error_msg = getattr(self, 'alignment_error', 'Unknown error occurred')
        
        # Provide helpful suggestions based on the error
        suggestions = "\n\nSuggestions:\n"
        if "size disparity" in error_msg.lower():
            suggestions += "• Try downsampling the larger point cloud\n"
            suggestions += "• Use the 'Downsample' button in the Preprocessing section\n"
        elif "memory" in error_msg.lower():
            suggestions += "• Reduce the point cloud size using downsampling\n"
            suggestions += "• Close other applications to free memory\n"
        elif "feature" in error_msg.lower() or "fpfh" in error_msg.lower():
            suggestions += "• Try disabling 'Use FPFH Features'\n"
            suggestions += "• Ensure point clouds have sufficient structure\n"
            suggestions += "• Try the 'Rigid' method instead\n"
        else:
            suggestions += "• Check that both datasets are properly loaded\n"
            suggestions += "• Try a different alignment method\n"
            suggestions += "• Consider preprocessing (outlier removal, downsampling)\n"
        
        messagebox.showerror("Alignment Failed", 
                           f"{error_msg}\n{suggestions}\n\nCheck the console for detailed error information.")
        
    def save_aligned(self):
        """Save aligned data"""
        if self.aligned_data is None:
            messagebox.showwarning("Warning", "No aligned data to save")
            return
            
        # Determine file types based on source type and aligned data type
        if self.source_type == 'mesh' or isinstance(self.aligned_data, trimesh.Trimesh):
            filetypes = [("PLY", "*.ply"), ("OBJ", "*.obj"), ("STL", "*.stl"), ("OFF", "*.off")]
            defaultext = ".ply"
            data_description = "mesh"
        else:
            filetypes = [("PLY", "*.ply"), ("PCD", "*.pcd"), ("XYZ", "*.xyz")]
            defaultext = ".ply"
            data_description = "point cloud"
            
        filename = filedialog.asksaveasfilename(
            title=f"Save Aligned {data_description.capitalize()}",
            defaultextension=defaultext,
            filetypes=filetypes + [("All files", "*.*")]
        )
        
        if filename:
            try:
                if isinstance(self.aligned_data, trimesh.Trimesh):
                    # Save mesh
                    self.aligned_data.export(filename)
                    print(f"Saved aligned mesh to: {filename}")
                    print(f"  Vertices: {len(self.aligned_data.vertices):,}")
                    print(f"  Faces: {len(self.aligned_data.faces):,}")
                else:
                    # Save point cloud
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(self.aligned_data['points'])
                    if self.aligned_data.get('normals') is not None:
                        pcd.normals = o3d.utility.Vector3dVector(self.aligned_data['normals'])
                    if self.aligned_data.get('colors') is not None:
                        pcd.colors = o3d.utility.Vector3dVector(self.aligned_data['colors'])
                    o3d.io.write_point_cloud(filename, pcd)
                    print(f"Saved aligned point cloud to: {filename}")
                    print(f"  Points: {len(self.aligned_data['points']):,}")
                    
                messagebox.showinfo("Success", f"Saved to {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {str(e)}")
                
    def downsample_dialog(self):
        """Show downsampling dialog"""
        if self.source_type != 'pointcloud' or self.source_data is None:
            messagebox.showinfo("Info", "Please load a source point cloud first")
            return
            
        dialog = tk.Toplevel(self.root)
        dialog.title("Downsample Point Cloud")
        dialog.geometry("300x150")
        
        ttk.Label(dialog, text="Voxel Size:").grid(row=0, column=0, padx=10, pady=10)
        voxel_var = tk.DoubleVar(value=0.01)
        ttk.Entry(dialog, textvariable=voxel_var).grid(row=0, column=1, padx=10, pady=10)
        
        def apply():
            voxel_size = voxel_var.get()
            points = self.source_data['points']
            normals = self.source_data.get('normals')
            colors = self.source_data.get('colors')
            
            if normals is not None or colors is not None:
                result = PointCloudProcessor.voxel_downsample(
                    points, voxel_size, normals, colors
                )
                if normals is not None and colors is not None:
                    points, normals, colors = result
                elif normals is not None:
                    points, normals = result
                else:
                    points, colors = result
            else:
                points = PointCloudProcessor.voxel_downsample(points, voxel_size)
                
            self.source_data = {'points': points, 'normals': normals, 'colors': colors}
            self.source_info.config(text=f"{len(points):,} points")
            self.update_stats()
            self.update_visualization()
            dialog.destroy()
            messagebox.showinfo("Success", f"Downsampled to {len(points):,} points")
            
        ttk.Button(dialog, text="Apply", command=apply).grid(row=1, column=0, columnspan=2, pady=20)
        
    def remove_outliers(self):
        """Remove outliers from point cloud"""
        if self.source_type != 'pointcloud' or self.source_data is None:
            messagebox.showinfo("Info", "Please load a source point cloud first")
            return
            
        points = self.source_data['points']
        cleaned_points, indices = PointCloudProcessor.statistical_outlier_removal(points)
        
        # Update normals and colors if present
        normals = self.source_data.get('normals')
        colors = self.source_data.get('colors')
        
        if normals is not None:
            normals = normals[indices]
        if colors is not None:
            colors = colors[indices]
            
        removed = len(points) - len(cleaned_points)
        self.source_data = {'points': cleaned_points, 'normals': normals, 'colors': colors}
        self.source_info.config(text=f"{len(cleaned_points):,} points")
        self.update_stats()
        self.update_visualization()
        
        messagebox.showinfo("Success", f"Removed {removed:,} outlier points")
        
    def estimate_normals(self):
        """Estimate normals for point cloud"""
        if self.source_type != 'pointcloud' or self.source_data is None:
            messagebox.showinfo("Info", "Please load a source point cloud first")
            return
            
        points = self.source_data['points']
        normals = PointCloudProcessor.estimate_normals(points)
        self.source_data['normals'] = normals
        
        self.update_stats()
        messagebox.showinfo("Success", "Normals estimated successfully")
        
    def convert_to_mesh(self):
        """Convert point cloud to mesh"""
        if self.source_type != 'pointcloud' or self.source_data is None:
            messagebox.showinfo("Info", "Please load a source point cloud first")
            return
            
        dialog = tk.Toplevel(self.root)
        dialog.title("Convert to Mesh")
        dialog.geometry("350x200")
        
        ttk.Label(dialog, text="Method:").grid(row=0, column=0, padx=10, pady=10)
        method_var = tk.StringVar(value="poisson")
        method_combo = ttk.Combobox(dialog, textvariable=method_var,
                                   values=["poisson", "ball_pivoting", "alpha"])
        method_combo.grid(row=0, column=1, padx=10, pady=10)
        
        ttk.Label(dialog, text="Depth (Poisson):").grid(row=1, column=0, padx=10, pady=10)
        depth_var = tk.IntVar(value=9)
        ttk.Spinbox(dialog, from_=6, to=12, textvariable=depth_var).grid(row=1, column=1, padx=10, pady=10)
        
        def apply():
            try:
                method = method_var.get()
                depth = depth_var.get()
                
                points = self.source_data['points']
                normals = self.source_data.get('normals')
                
                mesh = PointCloudProcessor.points_to_mesh(points, method, depth, normals)
                
                # Replace source with mesh
                self.source_data = mesh
                self.source_type = 'mesh'
                self.source_label.config(text="Mesh (converted)")
                self.source_info.config(text=f"{len(mesh.vertices):,} vertices")
                
                self.update_stats()
                self.update_visualization()
                dialog.destroy()
                
                messagebox.showinfo("Success", 
                    f"Converted to mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
                    
            except Exception as e:
                messagebox.showerror("Error", f"Conversion failed: {str(e)}")
                
        ttk.Button(dialog, text="Apply", command=apply).grid(row=2, column=0, columnspan=2, pady=20)
                
    def show_settings(self):
        """Show algorithm settings"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Algorithm Settings")
        settings_window.geometry("400x300")
        
        ttk.Label(settings_window, text="Advanced Algorithm Settings", 
                 font=("Arial", 12, "bold")).pack(pady=10)
        
        # Add settings controls here
        frame = ttk.Frame(settings_window, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="Max Iterations:").grid(row=0, column=0, sticky=tk.W)
        max_iter_var = tk.IntVar(value=self.aligner.max_iterations)
        ttk.Spinbox(frame, from_=10, to=200, textvariable=max_iter_var,
                   width=10).grid(row=0, column=1)
        
        ttk.Label(frame, text="Tolerance:").grid(row=1, column=0, sticky=tk.W)
        tol_var = tk.DoubleVar(value=self.aligner.tolerance)
        ttk.Entry(frame, textvariable=tol_var, width=10).grid(row=1, column=1)
        
        def apply():
            self.aligner.max_iterations = max_iter_var.get()
            self.aligner.tolerance = tol_var.get()
            settings_window.destroy()
            
        ttk.Button(frame, text="Apply", command=apply).grid(row=10, column=0, columnspan=2, pady=20)
        
    def show_about(self):
        """Show about dialog"""
        about_text = """Advanced Mesh & Point Cloud Alignment Tool
        
Version 3.0 - Hybrid Alignment System
        
Supported Formats:
• Meshes: PLY, OBJ, STL, OFF, GLB/GLTF
• Point Clouds: PLY, PCD, XYZ, PTS, CSV, LAS/LAZ
        
Alignment Methods:
• Rigid (ICP) - Fast rigid registration
• Non-rigid (CPD) - Coherent Point Drift
• Hybrid - Combined rigid + non-rigid
        
Features:
• Point-to-point alignment
• Point-to-mesh alignment  
• Mesh-to-mesh alignment
• FPFH feature matching
• Automatic normal estimation
• Outlier removal
• Voxel downsampling
• Point cloud to mesh conversion
        
Optimizations:
• GPU acceleration (if available)
• Parallel processing
• Adaptive sampling
• Multi-resolution pipeline
        
© 2025 - Advanced 3D Processing Suite"""
        
        messagebox.showinfo("About", about_text)
