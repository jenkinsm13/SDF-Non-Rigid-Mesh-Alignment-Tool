#!/usr/bin/env python3
"""
Main entry point for the SDF Non-Rigid Mesh Alignment Tool.
Initializes and runs the GUI application using the new package structure.
"""

import sys
import tkinter as tk
from tkinter import messagebox

from sdf_mesh_aligner.gui_app import main as gui_main
from sdf_mesh_aligner import print_info

def main():
    """Main entry point"""
    # Print library information
    print_info()
    
    # Run GUI application
    gui_main()

if __name__ == "__main__":
    main()
