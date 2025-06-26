"""
Utility functions for the GAN Image Generator
"""
import os
import sys
import importlib.util
from typing import List, Tuple
import re
import numpy as np


def monkey_patch_distutils() -> None:
    """Fix distutils compatibility issues for Python 3.12+"""
    try:
        import setuptools
        os.environ['SETUPTOOLS_USE_DISTUTILS'] = 'stdlib'
        
        try:
            import distutils
        except ImportError:
            pass
        
        # Fix _msvccompiler issues
        try:
            import distutils._msvccompiler
        except (ImportError, AttributeError):
            try:
                import setuptools._distutils._msvccompiler
                sys.modules['distutils._msvccompiler'] = setuptools._distutils._msvccompiler
            except ImportError:
                pass
        
        # Fix distutils.util issues
        try:
            import distutils.util
        except ImportError:
            try:
                import setuptools._distutils.util
                sys.modules['distutils.util'] = setuptools._distutils.util
            except ImportError:
                pass
                
    except ImportError:
        pass


def parse_range(range_string: str) -> List[int]:
    """
    Parse comma-separated numbers or ranges into a list of integers
    
    Args:
        range_string: String like "0,1,2" or "0-5" or "1,3-5,7"
    
    Returns:
        List of integers
    
    Examples:
        >>> parse_range("0,1,2")
        [0, 1, 2]
        >>> parse_range("0-3")
        [0, 1, 2, 3]
        >>> parse_range("1,3-5,7")
        [1, 3, 4, 5, 7]
    """
    if not range_string.strip():
        return []
    
    ranges = []
    range_pattern = re.compile(r'^(\d+)-(\d+)$')
    
    for part in range_string.split(','):
        part = part.strip()
        match = range_pattern.match(part)
        
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            ranges.extend(range(start, end + 1))
        else:
            ranges.append(int(part))
    
    return ranges


def make_transform_matrix(translate: Tuple[float, float], angle: float) -> np.ndarray:
    """
    Create transformation matrix for StyleGAN3
    
    Args:
        translate: (x, y) translation values
        angle: Rotation angle in degrees
    
    Returns:
        3x3 transformation matrix
    """
    matrix = np.eye(3)
    angle_rad = angle / 360.0 * np.pi * 2
    sin_val = np.sin(angle_rad)
    cos_val = np.cos(angle_rad)
    
    matrix[0][0] = cos_val
    matrix[0][1] = sin_val
    matrix[0][2] = translate[0]
    matrix[1][0] = -sin_val
    matrix[1][1] = cos_val
    matrix[1][2] = translate[1]
    
    return matrix


def get_available_models(model_dir: str) -> List[str]:
    """
    Get list of available model files in the specified directory
    
    Args:
        model_dir: Directory to search for .pkl files
    
    Returns:
        List of model filenames
    """
    if not os.path.exists(model_dir):
        return []
    
    return [f for f in os.listdir(model_dir) if f.endswith('.pkl')]


def detect_model_type(model_filename: str) -> str:
    """
    Detect model type from filename
    
    Args:
        model_filename: Name of the model file
    
    Returns:
        Model type string ("StyleGAN2-ADA" or "StyleGAN3")
    """
    filename_lower = model_filename.lower()
    
    if any(keyword in filename_lower for keyword in ['stylegan3', 'sg3']):
        return "StyleGAN3"
    else:
        return "StyleGAN2-ADA"


def clean_module_imports(module_keywords: List[str]) -> None:
    """
    Clean previously imported modules to avoid conflicts
    
    Args:
        module_keywords: List of keywords to match module names for removal
    """
    modules_to_remove = []
    
    for module_name in sys.modules:
        if any(keyword in module_name for keyword in module_keywords):
            modules_to_remove.append(module_name)
    
    for module_name in modules_to_remove:
        if module_name in sys.modules:
            del sys.modules[module_name]


def clean_sys_paths(path_keywords: List[str]) -> None:
    """
    Clean sys.path entries containing specific keywords
    
    Args:
        path_keywords: List of keywords to match path entries for removal
    """
    paths_to_remove = []
    
    for path in sys.path:
        if any(keyword in path.lower() for keyword in path_keywords):
            paths_to_remove.append(path)
    
    for path in paths_to_remove:
        if path in sys.path:
            sys.path.remove(path)
