"""Setup script for HydroAtmosFusion."""
from setuptools import setup, find_packages

setup(
    name="hydroatmosfusion",
    version="1.0.0",
    author="Jogi Panggabean, Noir Primadona Purba",
    author_email="jogipanggabeann@gmail.com",
    description="Multimodal deep learning for extreme rainfall prediction using ocean-atmosphere coupling",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jogipanggabean/HydroAtmosFusion",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scipy>=1.11.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "tensorboard>=2.14.0",
        "scikit-learn>=1.3.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
    keywords="deep-learning rainfall-prediction ocean-atmosphere multimodal pytorch",
)
