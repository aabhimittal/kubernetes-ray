"""Packaging for the Ray + Kubernetes GPU scaling project."""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
long_description = (ROOT / "README.md").read_text(encoding="utf-8")

setup(
    name="ray-k8s-gpu-scaling",
    version="0.1.0",
    description="Enterprise-grade distributed GPU training with Ray and Kubernetes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Abhishek Mittal",
    license="MIT",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
    ],
    extras_require={
        # Full runtime with GPU/cluster support.
        "full": [
            "ray[default]>=2.9.0",
            "torch>=2.1",
            "pynvml>=11.5",
            "prometheus-client>=0.19",
        ],
        # Development / testing.
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
