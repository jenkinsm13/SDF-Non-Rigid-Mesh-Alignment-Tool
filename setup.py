#!/usr/bin/env python3
"""
Setup script for SDF Non-Rigid Mesh Alignment Tool
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="sdf-mesh-aligner",
    version="2.0.0",
    author="Advanced Mesh Processing",
    author_email="contact@meshprocessing.com",
    description="Non-Rigid Mesh Alignment Tool using SDF-based energy minimization",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/sdf-mesh-aligner",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "gpu": ["cupy-cuda11x", "cupy-cuda12x"],
        "dev": ["pytest", "black", "flake8", "mypy"],
        "docs": ["sphinx", "sphinx-rtd-theme"],
    },
    entry_points={
        "console_scripts": [
            "sdf-aligner=sdf_mesh_aligner.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "sdf_mesh_aligner": ["*.json", "*.yaml", "*.yml"],
    },
    keywords="mesh alignment, SDF, non-rigid, 3D, computer vision, photogrammetry",
    project_urls={
        "Bug Reports": "https://github.com/your-repo/sdf-mesh-aligner/issues",
        "Source": "https://github.com/your-repo/sdf-mesh-aligner",
        "Documentation": "https://sdf-mesh-aligner.readthedocs.io/",
    },
)
