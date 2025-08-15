#!/usr/bin/env python3
"""
Configuration and settings management for SDF Mesh Aligner
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigManager:
    """Manages configuration settings for the mesh alignment tool"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "alignment_config.json"
        self.config = self.load_default_config()
        self.load_config()
        
    def load_default_config(self) -> Dict[str, Any]:
        """Load default configuration settings"""
        return {
            # Algorithm parameters
            "algorithm": {
                "lambda_smooth": 0.01,  # Lower default for hybrid alignment
                "lambda_detail": 0.1,   # Lower default for stronger deformation
                "lambda_rigid": 0.01,
                "sigma": 0.1,
                "max_iterations": 100,  # Increased default
                "tolerance": 1e-4,
                "subsample_ratio": 0.3,
            },
            
            # Point cloud parameters
            "point_cloud": {
                "use_fpfh": True,
                "use_icp_init": True,
                "correspondence_dist": 0.1,
                "auto_correspondence": True,
            },
            
            # Optimization settings
            "optimization": {
                "use_gpu": True,
                "use_octree": True,
                "use_sparse": True,
                "chunk_size": 10000,
                "cache_size": 1000000,
                "n_threads": os.cpu_count(),
            },
            
            # GUI settings
            "gui": {
                "window_width": 1400,
                "window_height": 900,
                "theme": "default",
                "auto_save": True,
                "show_progress": True,
                "log_level": "INFO",
            },
            
            # File settings
            "files": {
                "default_input_dir": "",
                "default_output_dir": "",
                "auto_backup": True,
                "backup_count": 5,
                "supported_formats": [".ply", ".obj", ".stl", ".off", ".glb", ".gltf", ".pcd", ".xyz", ".pts", ".txt", ".csv", ".las", ".laz"],
            },
            
            # Performance settings
            "performance": {
                "max_memory_gb": 8.0,
                "enable_caching": True,
                "cache_directory": "",
                "parallel_processing": True,
            },
            
            # Advanced settings
            "advanced": {
                "multi_resolution_levels": 3,
                "outlier_threshold": 0.95,
                "curvature_weight": 0.3,
                "edge_weight": 0.2,
                "adaptive_sampling": True,
                "early_stopping": True,
            }
        }
    
    def load_config(self) -> bool:
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    
                # Merge with defaults (preserve structure)
                self.merge_config(self.config, loaded_config)
                print(f"Configuration loaded from {self.config_file}")
                return True
            else:
                print(f"Configuration file not found: {self.config_file}")
                return False
                
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return False
    
    def save_config(self) -> bool:
        """Save configuration to file"""
        try:
            # Create directory if it doesn't exist
            config_dir = os.path.dirname(self.config_file)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir)
                
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
                
            print(f"Configuration saved to {self.config_file}")
            return True
            
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
    
    def merge_config(self, default: Dict[str, Any], loaded: Dict[str, Any]) -> None:
        """Recursively merge loaded configuration with defaults"""
        for key, value in loaded.items():
            if key in default:
                if isinstance(value, dict) and isinstance(default[key], dict):
                    self.merge_config(default[key], value)
                else:
                    default[key] = value
            else:
                default[key] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any) -> bool:
        """Set configuration value using dot notation"""
        keys = key_path.split('.')
        config = self.config
        
        try:
            # Navigate to the parent of the target key
            for key in keys[:-1]:
                if key not in config:
                    config[key] = {}
                config = config[key]
            
            # Set the value
            config[keys[-1]] = value
            return True
            
        except Exception as e:
            print(f"Error setting configuration: {e}")
            return False
    
    def get_algorithm_params(self) -> Dict[str, Any]:
        """Get algorithm parameters"""
        return self.config["algorithm"].copy()
    
    def set_algorithm_params(self, params: Dict[str, Any]) -> None:
        """Set algorithm parameters"""
        self.config["algorithm"].update(params)
    
    def get_optimization_params(self) -> Dict[str, Any]:
        """Get optimization parameters"""
        return self.config["optimization"].copy()
    
    def set_optimization_params(self, params: Dict[str, Any]) -> None:
        """Set optimization parameters"""
        self.config["optimization"].update(params)
    
    def get_gui_params(self) -> Dict[str, Any]:
        """Get GUI parameters"""
        return self.config["gui"].copy()
    
    def set_gui_params(self, params: Dict[str, Any]) -> None:
        """Set GUI parameters"""
        self.config["gui"].update(params)
    
    def validate_config(self) -> bool:
        """Validate configuration values"""
        try:
            # Validate algorithm parameters
            alg = self.config["algorithm"]
            if not (0.0 <= alg["lambda_smooth"] <= 10.0):
                return False
            if not (0.0 <= alg["lambda_detail"] <= 10.0):
                return False
            if not (0.0 <= alg["lambda_rigid"] <= 10.0):
                return False
            if not (0.001 <= alg["sigma"] <= 10.0):
                return False
            if not (1 <= alg["max_iterations"] <= 1000):
                return False
            if not (1e-8 <= alg["tolerance"] <= 1e-2):
                return False
            if not (0.01 <= alg["subsample_ratio"] <= 1.0):
                return False
            
            # Validate optimization parameters
            opt = self.config["optimization"]
            if not isinstance(opt["use_gpu"], bool):
                return False
            if not isinstance(opt["use_octree"], bool):
                return False
            if not isinstance(opt["use_sparse"], bool):
                return False
            if not (1000 <= opt["chunk_size"] <= 100000):
                return False
            if not (100000 <= opt["cache_size"] <= 10000000):
                return False
            if not (1 <= opt["n_threads"] <= 64):
                return False
            
            # Validate GUI parameters
            gui = self.config["gui"]
            if not (800 <= gui["window_width"] <= 3000):
                return False
            if not (600 <= gui["window_height"] <= 2000):
                return False
            if not isinstance(gui["auto_save"], bool):
                return False
            if not isinstance(gui["show_progress"], bool):
                return False
            
            return True
            
        except Exception:
            return False
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults"""
        self.config = self.load_default_config()
    
    def export_config(self, filename: str) -> bool:
        """Export configuration to file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            print(f"Error exporting configuration: {e}")
            return False
    
    def import_config(self, filename: str) -> bool:
        """Import configuration from file"""
        try:
            with open(filename, 'r') as f:
                imported_config = json.load(f)
            
            # Validate imported config
            if not self.validate_imported_config(imported_config):
                return False
            
            # Merge with current config
            self.merge_config(self.config, imported_config)
            return True
            
        except Exception as e:
            print(f"Error importing configuration: {e}")
            return False
    
    def validate_imported_config(self, config: Dict[str, Any]) -> bool:
        """Validate imported configuration structure"""
        required_sections = ["algorithm", "optimization", "gui", "files", "performance", "advanced"]
        
        for section in required_sections:
            if section not in config:
                print(f"Missing required section: {section}")
                return False
        
        return True
    
    def get_preset_config(self, preset_name: str) -> Optional[Dict[str, Any]]:
        """Get preset configuration"""
        presets = {
            "fast": {
                "algorithm": {
                    "max_iterations": 25,
                    "subsample_ratio": 0.2,
                    "tolerance": 1e-3,
                },
                "optimization": {
                    "chunk_size": 5000,
                    "use_sparse": True,
                }
            },
            "accurate": {
                "algorithm": {
                    "max_iterations": 100,
                    "subsample_ratio": 0.5,
                    "tolerance": 1e-5,
                },
                "optimization": {
                    "chunk_size": 20000,
                    "use_sparse": True,
                }
            },
            "memory_efficient": {
                "algorithm": {
                    "max_iterations": 50,
                    "subsample_ratio": 0.1,
                },
                "optimization": {
                    "chunk_size": 5000,
                    "cache_size": 500000,
                    "use_sparse": True,
                }
            },
            "gpu_optimized": {
                "optimization": {
                    "use_gpu": True,
                    "chunk_size": 15000,
                    "use_sparse": True,
                }
            }
        }
        
        return presets.get(preset_name)
    
    def apply_preset(self, preset_name: str) -> bool:
        """Apply preset configuration"""
        preset = self.get_preset_config(preset_name)
        if preset:
            self.merge_config(self.config, preset)
            return True
        return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for display"""
        return {
            "algorithm": {
                "smoothness_weight": self.config["algorithm"]["lambda_smooth"],
                "detail_preservation": self.config["algorithm"]["lambda_detail"],
                "max_iterations": self.config["algorithm"]["max_iterations"],
                "control_points_ratio": self.config["algorithm"]["subsample_ratio"],
            },
            "optimization": {
                "gpu_acceleration": self.config["optimization"]["use_gpu"],
                "sparse_matrices": self.config["optimization"]["use_sparse"],
                "octree_indexing": self.config["optimization"]["use_octree"],
                "chunk_size": self.config["optimization"]["chunk_size"],
            },
            "performance": {
                "max_memory_gb": self.config["performance"]["max_memory_gb"],
                "parallel_processing": self.config["performance"]["parallel_processing"],
                "caching_enabled": self.config["performance"]["enable_caching"],
            }
        }
