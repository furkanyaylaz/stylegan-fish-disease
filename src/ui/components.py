"""
Streamlit user interface components
"""
import streamlit as st
import os
from typing import List, Tuple, Optional
import zipfile
from io import BytesIO

from config.settings import Config
from ..utils.helpers import get_available_models, detect_model_type, parse_range


class UIComponents:
    """Handles Streamlit UI components and interactions"""
    
    def __init__(self):
        self.config = Config()
    
    def setup_page_config(self) -> None:
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title=self.config.PAGE_TITLE,
            layout=self.config.LAYOUT
        )
    
    def render_header(self) -> None:
        """Render main page header"""
        st.title("StyleGAN Image Generator")
        st.markdown("StyleGAN2-ADA ve StyleGAN3 modelleri ile yuksek kaliteli goruntuler uretin")
    
    def render_sidebar(self) -> Tuple[str, str, dict]:
        """
        Render sidebar with model selection and parameters
        
        Returns:
            Tuple of (model_path, model_type, generation_params)
        """
        with st.sidebar:
            st.header("Model Settings")
            
            # Model selection
            model_path, model_type = self._render_model_selection()
            
            if not model_path:
                return "", "", {}
            
            st.divider()
            
            # Generation parameters
            generation_params = self._render_generation_parameters(model_type)
            
            st.divider()
            
            # Generate button
            generation_params['generate_button'] = st.button("Generate Images")
            
            return model_path, model_type, generation_params
    
    def _render_model_selection(self) -> Tuple[str, str]:
        """Render model selection interface"""
        available_models = get_available_models(self.config.MODEL_DIR)
        
        if not available_models:
            st.warning(f"No .pkl files found in '{self.config.MODEL_DIR}' folder")
            st.info(f"Place your model files in '{self.config.MODEL_DIR}' folder to get started")
            return "", ""
        
        selected_model = st.selectbox(
            "Select Model File",
            options=[""] + available_models,
            help=f"Select from available models in {self.config.MODEL_DIR} folder"
        )
        
        if not selected_model:
            return "", ""
        
        model_path = os.path.join(self.config.MODEL_DIR, selected_model)
        
        # Auto-detect model type
        detected_type = detect_model_type(selected_model)
        default_index = self.config.MODEL_TYPES.index(detected_type)
        
        model_type = st.selectbox(
            "Model Type",
            self.config.MODEL_TYPES,
            index=default_index,
            help="Select model type (auto-detected from filename)"
        )
        
        st.success(f"Selected model: {selected_model}")
        
        if model_type == "StyleGAN3":
            st.info("StyleGAN3: Will automatically switch to CPU mode if CUDA extension issues occur")
        
        return model_path, model_type
    
    def _render_generation_parameters(self, model_type: str) -> dict:
        """Render generation parameter controls"""
        st.header("Generation Parameters")
        
        params = {}
        
        # Seeds
        params['seeds_input'] = st.text_input(
            "Seed Values",
            value="0,1,2,3,4",
            help="Example: 0,1,2,3 or 0-10 or 42"
        )
        
        # Truncation
        params['truncation_psi'] = st.slider(
            "Truncation Psi",
            min_value=0.0,
            max_value=2.0,
            value=self.config.DEFAULT_TRUNCATION_PSI,
            step=0.1,
            help="Lower values produce more average images, higher values more diverse"
        )
        
        # Noise mode
        params['noise_mode'] = st.selectbox(
            "Noise Mode",
            self.config.NOISE_MODES,
            index=self.config.NOISE_MODES.index(self.config.DEFAULT_NOISE_MODE),
            help="Select noise application mode"
        )
        
        # StyleGAN3 specific parameters
        if model_type == "StyleGAN3":
            params.update(self._render_stylegan3_parameters())
        
        return params
    
    def _render_stylegan3_parameters(self) -> dict:
        """Render StyleGAN3 specific transformation parameters"""
        st.subheader("StyleGAN3 Transform")
        
        params = {}
        
        col1, col2 = st.columns(2)
        with col1:
            params['translate_x'] = st.slider("Translate X", -1.0, 1.0, 0.0, 0.1)
            params['translate_y'] = st.slider("Translate Y", -1.0, 1.0, 0.0, 0.1)
        with col2:
            params['rotate'] = st.slider("Rotate (degrees)", -180.0, 180.0, 0.0, 5.0)
        
        return params
    
    def render_welcome_screen(self, available_models: List[str]) -> None:
        """Render welcome screen with instructions"""
        if available_models:
            st.info("Please select a model file from the left panel")
        else:
            st.info(f"Please place .pkl files in '{self.config.MODEL_DIR}' folder")
        
        st.markdown("""
        ### Usage Guide
        
        1. **Model File**: Select from available models in the data/models folder
        2. **Model Type**: Choose StyleGAN2-ADA or StyleGAN3 (auto-detected)
        3. **Seed Values**: Enter seeds for images to generate (e.g., 0,1,2 or 0-10)
        4. **Parameters**: Adjust truncation, noise mode, and other settings
        5. **Generate**: Click the button to generate images
        
        ### Tips
        - **Truncation Psi**: 0.5-0.8 range produces higher quality but more similar images
        - **Seeds**: Different seed values generate different images
        - **StyleGAN3**: Use transform parameters to translate/rotate generated images
        
        ### Model Files Folder
        - Place your model files in `data/models/` folder
        - Supported format: `.pkl` files
        - Both StyleGAN2-ADA and StyleGAN3 models are supported
        """)
        
        if available_models:
            st.markdown("### Available Model Files")
            for model in available_models:
                st.markdown(f"- `{model}`")
    
    def render_generation_interface(
        self,
        generated_images: List[Tuple[int, any]],
        is_generating: bool = False
    ) -> None:
        """Render the main generation interface"""
        if is_generating:
            return
        
        if not generated_images:
            return
        
        st.success(f"{len(generated_images)} images successfully generated!")
        
        # Display images
        st.header("Generated Images")
        
        # Grid layout
        for i in range(0, len(generated_images), self.config.COLS_PER_ROW):
            cols = st.columns(self.config.COLS_PER_ROW)
            for j in range(self.config.COLS_PER_ROW):
                if i + j < len(generated_images):
                    seed, img = generated_images[i + j]
                    with cols[j]:
                        st.image(img, caption=f"Seed: {seed}")
                        
                        # Download button
                        buf = BytesIO()
                        img.save(buf, format="PNG")
                        st.download_button(
                            label="Download",
                            data=buf.getvalue(),
                            file_name=f"seed_{seed:04d}.png",
                            mime="image/png",
                            key=f"download_{seed}"
                        )
        
        # Bulk download
        if len(generated_images) > 1:
            self._render_bulk_download(generated_images)
    
    def _render_bulk_download(self, generated_images: List[Tuple[int, any]]) -> None:
        """Render bulk download interface"""
        st.header("Bulk Download")
        
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for seed, img in generated_images:
                img_buffer = BytesIO()
                img.save(img_buffer, format="PNG")
                zip_file.writestr(f"seed_{seed:04d}.png", img_buffer.getvalue())
        
        st.download_button(
            label="Download All Images as ZIP",
            data=zip_buffer.getvalue(),
            file_name="generated_images.zip",
            mime="application/zip"
        )
    
    def show_progress(self, current: int, total: int, message: str = "") -> None:
        """Show progress bar and status"""
        progress = current / total
        st.progress(progress)
        if message:
            st.text(message)
    
    def validate_seeds(self, seeds_input: str) -> Optional[List[int]]:
        """Validate and parse seed input"""
        try:
            seeds = parse_range(seeds_input)
            if not seeds:
                st.error("Please enter valid seed values")
                return None
            return seeds
        except Exception:
            st.error("Invalid seed format. Example: 0,1,2 or 0-10")
            return None
    
    def show_error(self, message: str) -> None:
        """Display error message"""
        st.error(message)
