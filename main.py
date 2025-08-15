#!/usr/bin/env python3
"""
Main entry point for the SDF Non-Rigid Mesh Alignment Tool.
Initializes and runs the GUI application using the new package structure.
"""

import sys
import tkinter as tk
from tkinter import messagebox

from sdf_mesh_aligner import HybridAlignmentGUI, print_info

def main():
    """Main entry point"""
    # Print library information
    print_info()
    
    # Create and run the GUI
    root = tk.Tk()
    app = HybridAlignmentGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
