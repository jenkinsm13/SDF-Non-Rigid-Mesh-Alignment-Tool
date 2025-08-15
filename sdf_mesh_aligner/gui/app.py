#!/usr/bin/env python3
"""
Main GUI application for SDF Mesh Aligner
Combines original and optimized GUI features
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

from ..core.aligner import SDFMeshAligner, OptimizedSDFMeshAligner
from ..core.mesh_utils import MeshUtils
from ..config.settings import ConfigManager
from ..utils.performance import PerformanceMonitor
from ..utils.validation import validate_mesh, validate_config

# Try to import optional GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

class TextRedirector:
    """Redirect stdout to text widget"""
    
    def __init__(self, text_widget):
        self.text_widget = text_widget
        
    def write(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
        self.text_widget.update_idletasks()
        
    def flush(self):
        pass

class MeshAlignmentGUI:
    """Main GUI application for mesh alignment"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("SDF Non-Rigid Mesh Alignment Tool v2.0")
        self.root.geometry("1400x900")
        
        # Initialize components
        self.config_manager = ConfigManager()
        self.mesh_utils = MeshUtils()
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize meshes
        self.source_mesh = None
        self.target_mesh = None
        self.aligned_mesh = None
        
        # Initialize aligner (will be set based on mesh complexity)
        self.aligner = None
        
        # Setup GUI
        self.setup_menu()
        self.setup_widgets()
        
        # Redirect stdout to log
        sys.stdout = TextRedirector(self.log_text)
        
        # Print startup info
        self.print_startup_info()
        
    def setup_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Source Mesh", command=self.load_source_mesh)
        file_menu.add_command(label="Load Target Mesh", command=self.load_target_mesh)
        file_menu.add_separator()
        file_menu.add_command(label="Save Aligned Mesh", command=self.save_aligned_mesh)
        file_menu.add_command(label="Export Deformation Field", command=self.export_deformation)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="Algorithm Parameters", command=self.show_settings)
        settings_menu.add_command(label="Performance Options", command=self.show_performance_options)
        settings_menu.add_command(label="Presets", command=self.show_presets)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Mesh Statistics", command=self.show_mesh_stats)
        tools_menu.add_command(label="Alignment Report", command=self.generate_report)
        tools_menu.add_command(label="Performance Monitor", command=self.show_performance_monitor)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Performance Tips", command=self.show_performance_tips)
        help_menu.add_command(label="User Guide", command=self.show_user_guide)
        
    def setup_widgets(self):
        """Create main widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Mesh loading section
        load_frame = ttk.LabelFrame(control_frame, text="Mesh Loading", padding="5")
        load_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(load_frame, text="Load Source Mesh (Laser Scan)", 
                  command=self.load_source_mesh, width=30).grid(row=0, column=0, pady=5)
        ttk.Button(load_frame, text="Load Target Mesh (Photogrammetry)", 
                  command=self.load_target_mesh, width=30).grid(row=1, column=0, pady=5)
        
        # Status labels with mesh info
        self.source_label = ttk.Label(load_frame, text="No source mesh loaded", foreground="red")
        self.source_label.grid(row=0, column=1, padx=10)
        self.source_info = ttk.Label(load_frame, text="", font=("Courier", 9))
        self.source_info.grid(row=0, column=2, padx=5)
        
        self.target_label = ttk.Label(load_frame, text="No target mesh loaded", foreground="red")
        self.target_label.grid(row=1, column=1, padx=10)
        self.target_info = ttk.Label(load_frame, text="", font=("Courier", 9))
        self.target_info.grid(row=1, column=2, padx=5)
        
        # Algorithm parameters
        params_frame = ttk.LabelFrame(control_frame, text="Algorithm Parameters", padding="5")
        params_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Lambda smooth
        ttk.Label(params_frame, text="Smoothness (λ_smooth):").grid(row=0, column=0, sticky=tk.W)
        self.lambda_smooth_var = tk.DoubleVar(value=0.1)
        scale = ttk.Scale(params_frame, from_=0.001, to=1.0, variable=self.lambda_smooth_var, 
                         orient=tk.HORIZONTAL, length=150)
        scale.grid(row=0, column=1)
        ttk.Label(params_frame, textvariable=self.lambda_smooth_var).grid(row=0, column=2)
        
        # Lambda detail
        ttk.Label(params_frame, text="Detail Preservation (λ_detail):").grid(row=1, column=0, sticky=tk.W)
        self.lambda_detail_var = tk.DoubleVar(value=0.5)
        ttk.Scale(params_frame, from_=0.0, to=1.0, variable=self.lambda_detail_var,
                 orient=tk.HORIZONTAL, length=150).grid(row=1, column=1)
        ttk.Label(params_frame, textvariable=self.lambda_detail_var).grid(row=1, column=2)
        
        # Max iterations
        ttk.Label(params_frame, text="Max Iterations:").grid(row=2, column=0, sticky=tk.W)
        self.max_iter_var = tk.IntVar(value=50)
        ttk.Spinbox(params_frame, from_=10, to=200, textvariable=self.max_iter_var,
                   width=10).grid(row=2, column=1, sticky=tk.W)
        
        # Performance options
        perf_frame = ttk.LabelFrame(control_frame, text="Performance Options", padding="5")
        perf_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.use_gpu_var = tk.BooleanVar(value=GPU_AVAILABLE)
        ttk.Checkbutton(perf_frame, text=f"Use GPU Acceleration {'(Available)' if GPU_AVAILABLE else '(Not Available)'}", 
                       variable=self.use_gpu_var, 
                       state=tk.NORMAL if GPU_AVAILABLE else tk.DISABLED).grid(row=0, column=0, sticky=tk.W)
        
        self.use_octree_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(perf_frame, text="Use Octree Spatial Index", 
                       variable=self.use_octree_var).grid(row=0, column=1, sticky=tk.W)
        
        self.use_sparse_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(perf_frame, text="Use Sparse Matrices", 
                       variable=self.use_sparse_var).grid(row=1, column=0, sticky=tk.W)
        
        ttk.Label(perf_frame, text="Chunk Size:").grid(row=1, column=1, sticky=tk.W)
        self.chunk_size_var = tk.IntVar(value=10000)
        ttk.Spinbox(perf_frame, from_=1000, to=100000, increment=1000,
                   textvariable=self.chunk_size_var, width=10).grid(row=1, column=2)
        
        # Memory usage indicator
        self.memory_label = ttk.Label(perf_frame, text="Memory: 0 MB", font=("Courier", 9))
        self.memory_label.grid(row=2, column=0, columnspan=3, pady=5)
        self.update_memory_usage()
        
        # Align button
        self.align_button = ttk.Button(control_frame, text="ALIGN MESHES", 
                                      command=self.run_alignment, state=tk.DISABLED,
                                      style="Accent.TButton")
        self.align_button.grid(row=3, column=0, columnspan=2, pady=20)
        
        # Progress section
        progress_frame = ttk.Frame(control_frame)
        progress_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                           length=300, mode='determinate')
        self.progress_bar.pack(pady=5)
        
        self.progress_label = ttk.Label(progress_frame, text="")
        self.progress_label.pack()
        
        self.time_label = ttk.Label(progress_frame, text="", font=("Courier", 9))
        self.time_label.pack()
        
        # Save button
        self.save_button = ttk.Button(control_frame, text="Save Aligned Mesh", 
                                     command=self.save_aligned_mesh, state=tk.DISABLED)
        self.save_button.grid(row=5, column=0, columnspan=2, pady=10)
        
        # Visualization panel
        viz_frame = ttk.LabelFrame(main_frame, text="Mesh Visualization", padding="10")
        viz_frame.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Visualization controls
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
        
        # Create matplotlib figure
        self.fig = plt.figure(figsize=(10, 7))
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Log panel
        log_frame = ttk.LabelFrame(main_frame, text="Log Output", padding="10")
        log_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=12, width=50)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
    def print_startup_info(self):
        """Print startup information"""
        print("SDF Mesh Aligner v2.0 - Optimized for Large Meshes")
        print("="*50)
        print(f"GPU acceleration: {'Available' if GPU_AVAILABLE else 'Not available'}")
        print(f"CPU cores: {os.cpu_count()}")
        print(f"Configuration loaded: {self.config_manager.config_file}")
        
        if not GPU_AVAILABLE:
            print("For GPU acceleration, install CuPy: pip install cupy-cuda11x")
        
        print("="*50)
        
    def update_memory_usage(self):
        """Update memory usage display"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / (1024 * 1024)
            self.memory_label.config(text=f"Memory: {memory_mb:.1f} MB")
        except:
            pass
        self.root.after(1000, self.update_memory_usage)
        
    def load_source_mesh(self):
        """Load source mesh file"""
        filename = filedialog.askopenfilename(
            title="Select Source Mesh (Laser Scan)",
            filetypes=[("Mesh files", "*.ply *.obj *.stl *.off *.glb *.gltf"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.source_mesh = self.mesh_utils.load_mesh(filename)
                
                # Validate mesh
                is_valid, message = validate_mesh(self.source_mesh)
                if not is_valid:
                    messagebox.showerror("Validation Error", f"Source mesh validation failed: {message}")
                    return
                
                self.source_label.config(text=f"Loaded: {os.path.basename(filename)}", 
                                        foreground="green")
                
                mesh_info = self.mesh_utils.get_mesh_info(self.source_mesh)
                self.source_info.config(text=f"{mesh_info['vertices']:,} verts, "
                                           f"{mesh_info['faces']:,} faces")
                
                print(f"Loaded source mesh: {mesh_info['vertices']:,} vertices, "
                      f"{mesh_info['faces']:,} faces")
                
                # Get recommended settings
                recommendations = self.mesh_utils.get_recommended_settings(self.source_mesh)
                if recommendations['use_optimized']:
                    print("Large mesh detected. Optimization features enabled.")
                
                self.update_visualization()
                self.check_ready()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load source mesh: {str(e)}")
                
    def load_target_mesh(self):
        """Load target mesh file"""
        filename = filedialog.askopenfilename(
            title="Select Target Mesh (Photogrammetry)",
            filetypes=[("Mesh files", "*.ply *.obj *.stl *.off *.glb *.gltf"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.target_mesh = self.mesh_utils.load_mesh(filename)
                
                # Validate mesh
                is_valid, message = validate_mesh(self.target_mesh)
                if not is_valid:
                    messagebox.showerror("Validation Error", f"Target mesh validation failed: {message}")
                    return
                
                self.target_label.config(text=f"Loaded: {os.path.basename(filename)}", 
                                       foreground="green")
                
                mesh_info = self.mesh_utils.get_mesh_info(self.target_mesh)
                self.target_info.config(text=f"{mesh_info['vertices']:,} verts, "
                                           f"{mesh_info['faces']:,} faces")
                
                print(f"Loaded target mesh: {mesh_info['vertices']:,} vertices, "
                      f"{mesh_info['faces']:,} faces")
                
                self.update_visualization()
                self.check_ready()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load target mesh: {str(e)}")
                
    def check_ready(self):
        """Check if both meshes are loaded and ready for alignment"""
        if self.source_mesh is not None and self.target_mesh is not None:
            # Validate mesh pair
            is_valid, message = validate_mesh_pair(self.source_mesh, self.target_mesh)
            if not is_valid:
                messagebox.showwarning("Validation Warning", f"Mesh pair validation: {message}")
                return
            
            # Choose appropriate aligner
            recommendations = self.mesh_utils.get_recommended_settings(self.source_mesh)
            if recommendations['use_optimized']:
                self.aligner = OptimizedSDFMeshAligner()
                print("Using optimized aligner for large mesh")
            else:
                self.aligner = SDFMeshAligner()
                print("Using standard aligner")
            
            self.align_button.config(state=tk.NORMAL)
            
            # Estimate memory requirements
            total_vertices = len(self.source_mesh.vertices) + len(self.target_mesh.vertices)
            estimated_memory_gb = (total_vertices * 1000) / (1024**3)  # Rough estimate
            
            if estimated_memory_gb > 8.0:  # 8 GB threshold
                messagebox.showwarning("Memory Warning", 
                    f"Estimated memory requirement ({estimated_memory_gb:.1f} GB) "
                    f"may be high. Consider enabling optimization features.")
        else:
            self.align_button.config(state=tk.DISABLED)
            
    def update_visualization(self):
        """Update mesh visualization"""
        self.fig.clear()
        
        mode = self.display_mode.get()
        
        if mode == "all":
            # Show all meshes
            n_plots = sum([self.source_mesh is not None, 
                          self.target_mesh is not None, 
                          self.aligned_mesh is not None])
            
            if n_plots == 0:
                return
                
            plot_idx = 1
            
            if self.source_mesh is not None:
                ax = self.fig.add_subplot(1, n_plots, plot_idx, projection='3d')
                self.plot_mesh(ax, self.source_mesh, 'blue', "Source Mesh", sample_rate=100)
                plot_idx += 1
                
            if self.target_mesh is not None:
                ax = self.fig.add_subplot(1, n_plots, plot_idx, projection='3d')
                self.plot_mesh(ax, self.target_mesh, 'red', "Target Mesh", sample_rate=100)
                plot_idx += 1
                
            if self.aligned_mesh is not None:
                ax = self.fig.add_subplot(1, n_plots, plot_idx, projection='3d')
                self.plot_mesh(ax, self.aligned_mesh, 'green', "Aligned Mesh", sample_rate=100)
                
        elif mode == "overlay" and self.aligned_mesh is not None and self.target_mesh is not None:
            # Overlay visualization
            ax = self.fig.add_subplot(111, projection='3d')
            self.plot_mesh(ax, self.target_mesh, 'red', "Target", sample_rate=200, alpha=0.3)
            self.plot_mesh(ax, self.aligned_mesh, 'green', "Aligned", sample_rate=200, alpha=0.7)
            ax.set_title("Overlay: Target (red) vs Aligned (green)")
            
        else:
            # Single mesh view
            ax = self.fig.add_subplot(111, projection='3d')
            
            if mode == "source" and self.source_mesh is not None:
                self.plot_mesh(ax, self.source_mesh, 'blue', "Source Mesh", sample_rate=50)
            elif mode == "target" and self.target_mesh is not None:
                self.plot_mesh(ax, self.target_mesh, 'red', "Target Mesh", sample_rate=50)
            elif mode == "aligned" and self.aligned_mesh is not None:
                self.plot_mesh(ax, self.aligned_mesh, 'green', "Aligned Mesh", sample_rate=50)
                
        self.fig.tight_layout()
        self.canvas.draw()
        
    def plot_mesh(self, ax, mesh, color, title, sample_rate=100, alpha=0.5):
        """Plot mesh with adaptive sampling"""
        vertices = mesh.vertices
        
        # Adaptive sampling based on mesh size
        if len(vertices) > 10000:
            indices = np.random.choice(len(vertices), len(vertices)//sample_rate, replace=False)
            vertices = vertices[indices]
            
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                  c=color, s=1, alpha=alpha)
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set equal aspect ratio
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
        """Update progress bar"""
        self.progress_var.set(progress * 100)
        self.progress_label.config(text=message)
        self.root.update_idletasks()
        
    def run_alignment(self):
        """Run mesh alignment in separate thread"""
        if self.aligner is None:
            messagebox.showerror("Error", "No aligner initialized")
            return
        
        # Update algorithm parameters
        self.aligner.lambda_smooth = self.lambda_smooth_var.get()
        self.aligner.lambda_detail = self.lambda_detail_var.get()
        self.aligner.max_iterations = self.max_iter_var.get()
        
        # Update optimization parameters
        if hasattr(self.aligner, 'use_gpu'):
            self.aligner.use_gpu = self.use_gpu_var.get() and GPU_AVAILABLE
            self.aligner.use_octree = self.use_octree_var.get()
            self.aligner.use_sparse = self.use_sparse_var.get()
            self.aligner.chunk_size = self.chunk_size_var.get()
        
        # Disable controls during alignment
        self.align_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        
        # Start performance monitoring
        self.performance_monitor.start_monitoring()
        
        # Start timer
        self.start_time = time.time()
        self.update_timer()
        
        # Run alignment in thread
        thread = threading.Thread(target=self.alignment_worker)
        thread.start()
        
    def update_timer(self):
        """Update elapsed time display"""
        if hasattr(self, 'start_time') and self.align_button['state'] == str(tk.DISABLED):
            elapsed = time.time() - self.start_time
            self.time_label.config(text=f"Elapsed: {elapsed:.1f}s")
            self.root.after(100, self.update_timer)
            
    def alignment_worker(self):
        """Worker thread for alignment"""
        try:
            # Run alignment
            self.aligned_mesh, final_energy = self.aligner.align(
                self.source_mesh, 
                self.target_mesh,
                progress_callback=self.progress_callback
            )
            
            # Stop performance monitoring
            self.performance_monitor.stop_monitoring()
            
            elapsed_time = time.time() - self.start_time
            
            # Compute alignment metrics
            metrics = self.mesh_utils.compute_alignment_metrics(
                self.source_mesh, self.target_mesh, self.aligned_mesh
            )
            
            print(f"\n{'='*50}")
            print(f"ALIGNMENT COMPLETE")
            print(f"{'='*50}")
            print(f"Time: {elapsed_time:.2f} seconds")
            print(f"Final energy: {final_energy:.6f}")
            print(f"Average distance to target: {metrics['mean_distance']:.6f}")
            print(f"Maximum distance to target: {metrics['max_distance']:.6f}")
            print(f"Std deviation: {metrics['std_distance']:.6f}")
            print(f"Vertices processed: {len(self.aligned_mesh.vertices):,}")
            print(f"{'='*50}")
            
            # Update GUI in main thread
            self.root.after(0, self.alignment_complete)
            
        except Exception as e:
            print(f"Alignment failed: {str(e)}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Alignment failed: {str(e)}")
            self.root.after(0, self.alignment_failed)
            
    def alignment_complete(self):
        """Called when alignment is complete"""
        self.align_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.NORMAL)
        self.update_visualization()
        
        # Show performance summary
        self.performance_monitor.print_summary()
        
        messagebox.showinfo("Success", "Mesh alignment completed successfully!")
        
    def alignment_failed(self):
        """Called when alignment fails"""
        self.align_button.config(state=tk.NORMAL)
        self.progress_var.set(0)
        self.progress_label.config(text="Alignment failed")
        
    def save_aligned_mesh(self):
        """Save aligned mesh to file"""
        if self.aligned_mesh is None:
            messagebox.showwarning("Warning", "No aligned mesh to save")
            return
            
        filename = filedialog.asksaveasfilename(
            title="Save Aligned Mesh",
            defaultextension=".ply",
            filetypes=[("PLY files", "*.ply"), ("OBJ files", "*.obj"), 
                      ("STL files", "*.stl"), ("GLTF files", "*.glb"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                success = self.mesh_utils.save_mesh(self.aligned_mesh, filename)
                if success:
                    print(f"Saved aligned mesh to: {filename}")
                    messagebox.showinfo("Success", f"Mesh saved to {filename}")
                else:
                    messagebox.showerror("Error", "Failed to save mesh")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save mesh: {str(e)}")
                
    def export_deformation(self):
        """Export deformation field"""
        if self.aligned_mesh is None:
            messagebox.showwarning("Warning", "No alignment data to export")
            return
            
        filename = filedialog.asksaveasfilename(
            title="Export Deformation Field",
            defaultextension=".npz",
            filetypes=[("NumPy files", "*.npz"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                deformation = self.aligned_mesh.vertices - self.source_mesh.vertices
                np.savez(filename, 
                        deformation=deformation,
                        source_vertices=self.source_mesh.vertices,
                        aligned_vertices=self.aligned_mesh.vertices)
                messagebox.showinfo("Success", f"Deformation field saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {str(e)}")
    
    def show_settings(self):
        """Show algorithm settings dialog"""
        # Implementation for settings dialog
        messagebox.showinfo("Settings", "Settings dialog coming soon...")
    
    def show_performance_options(self):
        """Show performance optimization options"""
        # Implementation for performance options dialog
        messagebox.showinfo("Performance", "Performance options dialog coming soon...")
    
    def show_presets(self):
        """Show preset configurations"""
        # Implementation for presets dialog
        messagebox.showinfo("Presets", "Presets dialog coming soon...")
    
    def show_mesh_stats(self):
        """Show detailed mesh statistics"""
        if not self.source_mesh and not self.target_mesh:
            messagebox.showwarning("Warning", "No meshes loaded")
            return
            
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Mesh Statistics")
        stats_window.geometry("600x400")
        
        text = scrolledtext.ScrolledText(stats_window, wrap=tk.WORD)
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text.insert(tk.END, "MESH STATISTICS\n" + "="*50 + "\n\n")
        
        if self.source_mesh:
            mesh_info = self.mesh_utils.get_mesh_info(self.source_mesh)
            text.insert(tk.END, "SOURCE MESH:\n")
            for key, value in mesh_info.items():
                text.insert(tk.END, f"  {key}: {value}\n")
            text.insert(tk.END, "\n")
            
        if self.target_mesh:
            mesh_info = self.mesh_utils.get_mesh_info(self.target_mesh)
            text.insert(tk.END, "TARGET MESH:\n")
            for key, value in mesh_info.items():
                text.insert(tk.END, f"  {key}: {value}\n")
            text.insert(tk.END, "\n")
            
        if self.aligned_mesh:
            mesh_info = self.mesh_utils.get_mesh_info(self.aligned_mesh)
            text.insert(tk.END, "ALIGNED MESH:\n")
            for key, value in mesh_info.items():
                text.insert(tk.END, f"  {key}: {value}\n")
    
    def generate_report(self):
        """Generate detailed alignment report"""
        if self.aligned_mesh is None:
            messagebox.showwarning("Warning", "No alignment data available")
            return
            
        filename = filedialog.asksaveasfilename(
            title="Save Alignment Report",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write("MESH ALIGNMENT REPORT\n")
                    f.write("="*50 + "\n\n")
                    f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    
                    # Add detailed metrics
                    metrics = self.mesh_utils.compute_alignment_metrics(
                        self.source_mesh, self.target_mesh, self.aligned_mesh
                    )
                    
                    f.write("ALIGNMENT METRICS:\n")
                    for key, value in metrics.items():
                        f.write(f"  {key}: {value:.6f}\n")
                    
                    # Add performance summary
                    perf_summary = self.performance_monitor.get_performance_summary()
                    f.write(f"\nPERFORMANCE:\n")
                    f.write(f"  Total time: {perf_summary['total_time']:.2f} seconds\n")
                    f.write(f"  Memory usage: {perf_summary['memory_usage']['rss_mb']:.1f} MB\n")
                    
                messagebox.showinfo("Success", f"Report saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to generate report: {str(e)}")
    
    def show_performance_monitor(self):
        """Show performance monitoring window"""
        self.performance_monitor.print_summary()
    
    def show_performance_tips(self):
        """Show performance tips"""
        tips = """Performance Tips for Large Meshes:

1. **Enable GPU Acceleration**: If you have a CUDA-capable GPU, install CuPy:
   pip install cupy-cuda11x

2. **Use Sparse Matrices**: Enabled by default for meshes > 10k vertices

3. **Adjust Chunk Size**: Larger chunks use more memory but run faster

4. **Reduce Control Points**: Lower subsample ratio for faster but less accurate alignment

5. **Multi-resolution**: Algorithm automatically uses 3 resolution levels

6. **Memory Management**: 
   - Close other applications
   - Increase system swap/page file
   - Use 64-bit Python

7. **Mesh Preprocessing**:
   - Decimate extremely dense meshes
   - Remove duplicate vertices
   - Clean mesh topology

8. **Optimal Settings by Mesh Size**:
   - < 10k vertices: Default settings
   - 10k-100k: Enable all optimizations
   - 100k-1M: Reduce control points, increase chunk size
   - > 1M: Consider mesh decimation first
"""
        
        messagebox.showinfo("Performance Tips", tips)
    
    def show_user_guide(self):
        """Show user guide"""
        guide = """SDF Mesh Aligner User Guide

1. **Loading Meshes**:
   - Load source mesh (laser scan) first
   - Load target mesh (photogrammetry) second
   - Supported formats: PLY, OBJ, STL, OFF, GLB, GLTF

2. **Algorithm Parameters**:
   - Smoothness: Controls surface smoothness (0.001-1.0)
   - Detail Preservation: Preserves fine details (0.0-1.0)
   - Max Iterations: Optimization iterations (10-200)

3. **Performance Options**:
   - GPU Acceleration: Use CUDA if available
   - Octree Indexing: Faster spatial queries
   - Sparse Matrices: Memory efficient for large meshes
   - Chunk Size: Memory vs speed trade-off

4. **Running Alignment**:
   - Click "ALIGN MESHES" to start
   - Monitor progress in real-time
   - View results in visualization panel

5. **Saving Results**:
   - Save aligned mesh in preferred format
   - Export deformation field for analysis
   - Generate detailed reports

6. **Troubleshooting**:
   - Check mesh validation messages
   - Monitor memory usage
   - Use performance tips for large meshes
"""
        
        messagebox.showinfo("User Guide", guide)
        
    def show_about(self):
        """Show about dialog"""
        about_text = """SDF Non-Rigid Mesh Alignment Tool v2.0
        
Optimized for Large Meshes
        
Features:
• Handles meshes with 100k+ vertices efficiently
• GPU acceleration support (CUDA)
• Sparse matrix operations
• Hierarchical optimization
• Adaptive sampling strategies
• Memory-efficient processing
        
Algorithm:
• SDF-based energy minimization
• Multi-resolution optimization
• Automatic outlier rejection
• Detail preservation
        
Performance:
• Parallel processing
• Octree spatial indexing
• Batch processing
• Intelligent caching
        
© 2024 - Advanced Mesh Processing"""
        
        messagebox.showinfo("About", about_text)

def main():
    """Main entry point for GUI application"""
    print("Starting SDF Mesh Alignment Tool GUI...")
    
    root = tk.Tk()
    
    # Set style
    style = ttk.Style()
    style.configure("Accent.TButton", foreground="blue", font=("Arial", 10, "bold"))
    
    app = MeshAlignmentGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
