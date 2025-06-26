"""
Configuration settings for GAN Image Generator
"""
import os
from typing import Dict, Any


class Config:
    """Main configuration class"""
    
    # Environment settings
    FORCE_CUDA = True
    TORCH_CUDA_ARCH_LIST = "6.0;6.1;7.0;7.5;8.0;8.6"
    DISTUTILS_USE_SDK = True
    SETUPTOOLS_USE_DISTUTILS = "stdlib"
    
    # Model settings
    DEFAULT_MODEL_TYPE = "StyleGAN2-ADA"
    MODEL_TYPES = ["StyleGAN2-ADA", "StyleGAN3"]
    
    # Generation settings
    DEFAULT_TRUNCATION_PSI = 1.0
    DEFAULT_NOISE_MODE = "const"
    NOISE_MODES = ["const", "random", "none"]
    
    # UI settings
    PAGE_TITLE = "StyleGAN Image Generator"
    LAYOUT = "wide"
    COLS_PER_ROW = 3
    
    # File paths
    MODEL_DIR = "data/models"
    METRICS_DIR = "data/metrics"
    OUTPUT_DIR = "outputs"
    
    @classmethod
    def setup_environment(cls) -> None:
        """Setup environment variables"""
        if cls.FORCE_CUDA:
            os.environ['FORCE_CUDA'] = '1'
        os.environ['TORCH_CUDA_ARCH_LIST'] = cls.TORCH_CUDA_ARCH_LIST
        os.environ['DISTUTILS_USE_SDK'] = '1'
        os.environ['SETUPTOOLS_USE_DISTUTILS'] = cls.SETUPTOOLS_USE_DISTUTILS
    
    @classmethod
    def get_model_config(cls, model_type: str) -> Dict[str, Any]:
        """Get configuration for specific model type"""
        configs = {
            "StyleGAN2-ADA": {
                "path": "stylegan2-ada-pytorch",
                "supports_transform": False,
                "default_resolution": 256
            },
            "StyleGAN3": {
                "path": "stylegan3",
                "supports_transform": True,
                "default_resolution": 512
            }
        }
        return configs.get(model_type, configs["StyleGAN2-ADA"])
