"""
Model loader and manager for StyleGAN models
"""
import sys
import os
from typing import Optional, Tuple, Any
import torch
import streamlit as st

from ..utils.helpers import clean_module_imports, clean_sys_paths
from config.settings import Config


class ModelManager:
    """Manages StyleGAN model loading and operations"""
    
    def __init__(self):
        self.dnnlib = None
        self.legacy = None
        self.current_model_type = None
        
    def import_stylegan_modules(self, model_type: str) -> bool:
        """
        Import appropriate StyleGAN modules based on model type
        
        Args:
            model_type: Type of StyleGAN model ("StyleGAN2-ADA" or "StyleGAN3")
        
        Returns:
            True if successful, False otherwise
        """
        # Clean previous imports
        clean_module_imports(['dnnlib', 'legacy', 'torch_utils'])
        clean_sys_paths(['stylegan'])
        
        try:
            model_config = Config.get_model_config(model_type)
            sys.path.insert(0, model_config["path"])
            
            import dnnlib as dnnlib_module
            import legacy as legacy_module
            
            self.dnnlib = dnnlib_module
            self.legacy = legacy_module
            self.current_model_type = model_type
            
            return True
            
        except ImportError as e:
            st.error(f"StyleGAN modules could not be loaded: {e}")
            return False
    
    def load_model(self, network_pkl: str, model_type: str) -> Tuple[Optional[Any], Optional[torch.device]]:
        """
        Load StyleGAN model from pickle file
        
        Args:
            network_pkl: Path to the model pickle file
            model_type: Type of StyleGAN model
        
        Returns:
            Tuple of (model, device) or (None, None) if failed
        """
        try:
            # Import correct modules
            if not self.import_stylegan_modules(model_type):
                return None, None
            
            # Determine device
            device = self._get_device()
            st.info(f"{model_type} model successfully loaded ({device})")
            
            # Load model
            with self.dnnlib.util.open_url(network_pkl) as f:
                G = self.legacy.load_network_pkl(f)['G_ema'].to(device)
            
            return G, device
            
        except Exception as e:
            self._handle_loading_error(e)
            return None, None
    
    def _get_device(self) -> torch.device:
        """
        Get appropriate torch device, handling distutils errors
        
        Returns:
            torch.device object
        """
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return device
        except Exception as device_error:
            if "distutils" in str(device_error) or "_msvccompiler" in str(device_error):
                st.warning("CUDA loading failed, running in CPU mode")
                return torch.device('cpu')
            else:
                raise device_error
    
    def _handle_loading_error(self, error: Exception) -> None:
        """
        Handle model loading errors with appropriate messages
        
        Args:
            error: Exception that occurred during loading
        """
        error_str = str(error)
        
        if "distutils" in error_str or "_msvccompiler" in error_str:
            st.error("distutils compatibility error: Please use Python 3.11 or older, or update setuptools")
            st.code("pip install --upgrade setuptools")
        else:
            st.error(f"Error loading model: {error}")
    
    def is_loaded(self) -> bool:
        """Check if modules are loaded"""
        return self.dnnlib is not None and self.legacy is not None
