"""
Image generation functionality for StyleGAN models
"""
from typing import Optional, Tuple
import numpy as np
import torch
import PIL.Image
import streamlit as st

from ..utils.helpers import make_transform_matrix


class ImageGenerator:
    """Handles image generation for StyleGAN models"""
    
    def __init__(self, model, device: torch.device, model_type: str):
        self.model = model
        self.device = device
        self.model_type = model_type
        self.is_stylegan3 = model_type == "StyleGAN3"
    
    def generate_single_image(
        self,
        seed: int,
        truncation_psi: float = 1.0,
        noise_mode: str = "const",
        class_idx: Optional[int] = None,
        translate: Tuple[float, float] = (0, 0),
        rotate: float = 0
    ) -> Optional[PIL.Image.Image]:
        """
        Generate a single image using the loaded model
        
        Args:
            seed: Random seed for generation
            truncation_psi: Truncation parameter
            noise_mode: Noise mode ("const", "random", "none")
            class_idx: Class index for conditional models
            translate: Translation parameters for StyleGAN3
            rotate: Rotation angle for StyleGAN3
        
        Returns:
            Generated PIL Image or None if failed
        """
        try:
            if self.is_stylegan3:
                return self._generate_stylegan3_image(
                    seed, truncation_psi, noise_mode, class_idx, translate, rotate
                )
            else:
                return self._generate_stylegan2_image(
                    seed, truncation_psi, noise_mode, class_idx
                )
        
        except Exception as e:
            st.error(f"Error generating image: {e}")
            import traceback
            st.error(f"Detailed error: {traceback.format_exc()}")
            return None
    
    def _generate_stylegan2_image(
        self,
        seed: int,
        truncation_psi: float,
        noise_mode: str,
        class_idx: Optional[int]
    ) -> Optional[PIL.Image.Image]:
        """Generate image using StyleGAN2-ADA"""
        # Prepare label
        label = torch.zeros([1, self.model.c_dim], device=self.device)
        if self.model.c_dim != 0 and class_idx is not None:
            label[:, class_idx] = 1
        
        # Generate random latent
        z = torch.from_numpy(
            np.random.RandomState(seed).randn(1, self.model.z_dim)
        ).to(self.device)
        
        # Generate image
        img = self.model(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        
        return PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
    
    def _generate_stylegan3_image(
        self,
        seed: int,
        truncation_psi: float,
        noise_mode: str,
        class_idx: Optional[int],
        translate: Tuple[float, float],
        rotate: float
    ) -> Optional[PIL.Image.Image]:
        """Generate image using StyleGAN3 with error handling for CUDA extensions"""
        try:
            return self._stylegan3_generation_attempt(
                seed, truncation_psi, noise_mode, class_idx, translate, rotate
            )
        
        except Exception as sg3_error:
            if self._is_cuda_extension_error(sg3_error):
                st.warning("StyleGAN3 CUDA extension error detected. Trying CPU mode...")
                return self._stylegan3_cpu_fallback(
                    seed, truncation_psi, noise_mode, class_idx, translate, rotate
                )
            else:
                raise sg3_error
    
    def _stylegan3_generation_attempt(
        self,
        seed: int,
        truncation_psi: float,
        noise_mode: str,
        class_idx: Optional[int],
        translate: Tuple[float, float],
        rotate: float
    ) -> PIL.Image.Image:
        """Attempt StyleGAN3 generation with transforms"""
        # Prepare label
        label = torch.zeros([1, self.model.c_dim], device=self.device)
        if self.model.c_dim != 0 and class_idx is not None:
            label[:, class_idx] = 1
        
        # Generate random latent
        z = torch.from_numpy(
            np.random.RandomState(seed).randn(1, self.model.z_dim)
        ).to(self.device)
        
        # Apply transform for StyleGAN3
        if hasattr(self.model.synthesis, 'input'):
            transform_matrix = make_transform_matrix(translate, rotate)
            transform_matrix = np.linalg.inv(transform_matrix)
            self.model.synthesis.input.transform.copy_(torch.from_numpy(transform_matrix))
        
        # Generate image
        img = self.model(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        
        return PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
    
    def _stylegan3_cpu_fallback(
        self,
        seed: int,
        truncation_psi: float,
        noise_mode: str,
        class_idx: Optional[int],
        translate: Tuple[float, float],
        rotate: float
    ) -> PIL.Image.Image:
        """Fallback to CPU generation for StyleGAN3"""
        # Move model to CPU
        cpu_device = torch.device('cpu')
        model_cpu = self.model.to(cpu_device)
        
        # Prepare label
        label = torch.zeros([1, model_cpu.c_dim], device=cpu_device)
        if model_cpu.c_dim != 0 and class_idx is not None:
            label[:, class_idx] = 1
        
        # Generate random latent
        z = torch.from_numpy(
            np.random.RandomState(seed).randn(1, model_cpu.z_dim)
        ).to(cpu_device)
        
        # Apply transform
        if hasattr(model_cpu.synthesis, 'input'):
            transform_matrix = make_transform_matrix(translate, rotate)
            transform_matrix = np.linalg.inv(transform_matrix)
            model_cpu.synthesis.input.transform.copy_(torch.from_numpy(transform_matrix))
        
        # Generate on CPU
        with torch.no_grad():
            img = model_cpu(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        
        return PIL.Image.fromarray(img[0].numpy(), 'RGB')
    
    def _is_cuda_extension_error(self, error: Exception) -> bool:
        """Check if error is related to CUDA extensions"""
        error_keywords = [
            'bias_act_plugin', 'upfirdn2d_plugin', 'extension', 
            'ninja', 'cl.exe', 'cstddef'
        ]
        return any(keyword in str(error).lower() for keyword in error_keywords)
