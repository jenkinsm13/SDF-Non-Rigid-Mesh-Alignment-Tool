#!/usr/bin/env python3
"""
GUI Application Entry Point for SDF Mesh Aligner
"""

import sys
import tkinter as tk
from tkinter import messagebox

from .gui.hybrid_gui import HybridAlignmentGUI
from .config.settings import ConfigManager
from .utils.validation import check_system_requirements

def check_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import trimesh
    except ImportError:
        missing_deps.append("trimesh")
    
    try:
        import scipy
    except ImportError:
        missing_deps.append("scipy")
    
    try:
        import matplotlib
    except ImportError:
        missing_deps.append("matplotlib")
    
    if missing_deps:
        error_msg = f"Missing required dependencies: {', '.join(missing_deps)}\n\n"
        error_msg += "Please install them using pip:\n"
        error_msg += f"pip install {' '.join(missing_deps)}"
        messagebox.showerror("Missing Dependencies", error_msg)
        return False
    
    return True

def main():
    """Main entry point for GUI application"""
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check system requirements
    meets_requirements, req_message = check_system_requirements()
    if not meets_requirements:
        messagebox.showerror("System Requirements", f"System requirements not met: {req_message}")
        sys.exit(1)
    
    # Load configuration
    config_manager = ConfigManager()
    
    # Validate configuration
    if not config_manager.validate_config():
        messagebox.showwarning("Configuration Warning", 
                              "Some configuration values are invalid. Using defaults.")
        config_manager.reset_to_defaults()
    
    try:
        # Create main window
        root = tk.Tk()
        
        # Create application
        app = HybridAlignmentGUI(root)
        
        # Start GUI event loop
        root.mainloop()
        
    except Exception as e:
        error_msg = f"Application failed to start: {str(e)}"
        messagebox.showerror("Startup Error", error_msg)
        sys.exit(1)

if __name__ == "__main__":
    main()
