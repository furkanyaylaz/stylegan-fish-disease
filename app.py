"""
Main Streamlit application for GAN Image Generator
Clean architecture implementation with proper separation of concerns
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import configuration and setup
from config.settings import Config
from src.utils.helpers import monkey_patch_distutils
from src.generators.model_manager import ModelManager
from src.generators.image_generator import ImageGenerator
from src.ui.components import UIComponents


class GanImageGeneratorApp:
    """Main application class for GAN Image Generator"""
    
    def __init__(self):
        self.config = Config()
        self.ui = UIComponents()
        self.model_manager = ModelManager()
        self.image_generator = None
        
        # Setup environment
        self._setup_environment()
    
    def _setup_environment(self) -> None:
        """Setup application environment"""
        monkey_patch_distutils()
        self.config.setup_environment()
    
    def run(self) -> None:
        """Main application entry point"""
        # Setup page configuration
        self.ui.setup_page_config()
        
        # Render header
        self.ui.render_header()
        
        # Get UI inputs
        model_path, model_type, generation_params = self.ui.render_sidebar()
        
        # Handle different application states
        if not model_path:
            available_models = self._get_available_models()
            self.ui.render_welcome_screen(available_models)
            return
        
        if not os.path.exists(model_path):
            self.ui.show_error(f"Model file not found: {model_path}")
            return
        
        # Handle generation request
        if generation_params.get('generate_button', False):
            self._handle_generation_request(model_path, model_type, generation_params)
    
    def _get_available_models(self) -> list:
        """Get list of available model files"""
        from src.utils.helpers import get_available_models
        return get_available_models(self.config.MODEL_DIR)
    
    def _handle_generation_request(
        self,
        model_path: str,
        model_type: str,
        generation_params: dict
    ) -> None:
        """Handle image generation request"""
        # Validate seeds
        seeds = self.ui.validate_seeds(generation_params['seeds_input'])
        if not seeds:
            return
        
        # Load model
        if not self._load_model(model_path, model_type):
            return
        
        # Generate images
        generated_images = self._generate_images(seeds, generation_params, model_type)
        
        # Display results
        self.ui.render_generation_interface(generated_images)
    
    def _load_model(self, model_path: str, model_type: str) -> bool:
        """Load the specified model"""
        import streamlit as st
        
        with st.spinner("Loading model..."):
            model, device = self.model_manager.load_model(model_path, model_type)
        
        if model is None:
            return False
        
        self.image_generator = ImageGenerator(model, device, model_type)
        return True
    
    def _generate_images(
        self,
        seeds: list,
        generation_params: dict,
        model_type: str
    ) -> list:
        """Generate images with the loaded model"""
        import streamlit as st
        
        generated_images = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, seed in enumerate(seeds):
            status_text.text(f"Generating image: {seed} ({i+1}/{len(seeds)})")
            progress_bar.progress((i + 1) / len(seeds))
            
            # Prepare generation parameters
            gen_kwargs = {
                'seed': seed,
                'truncation_psi': generation_params['truncation_psi'],
                'noise_mode': generation_params['noise_mode'],
                'class_idx': None
            }
            
            # Add StyleGAN3 specific parameters
            if model_type == "StyleGAN3":
                gen_kwargs['translate'] = (
                    generation_params.get('translate_x', 0),
                    generation_params.get('translate_y', 0)
                )
                gen_kwargs['rotate'] = generation_params.get('rotate', 0)
            
            # Generate image
            img = self.image_generator.generate_single_image(**gen_kwargs)
            
            if img:
                generated_images.append((seed, img))
        
        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()
        
        if generated_images:
            st.success(f"{len(generated_images)} images successfully generated!")
        
        return generated_images


def main():
    """Application entry point"""
    app = GanImageGeneratorApp()
    app.run()


if __name__ == "__main__":
    main()
