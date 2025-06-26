"""
LoRA Inference Manager for Stable Diffusion Models
Clean implementation for batch inference with fine-tuned models
"""
import os
import sys
import argparse
import json
import torch
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
from io import BytesIO

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import PeftModel

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class DiffusionInferenceConfig:
    """Configuration for diffusion model inference"""
    
    # Default model settings
    DEFAULT_MODEL_NAME = "runwayml/stable-diffusion-v1-5"
    DEFAULT_HEIGHT = 512
    DEFAULT_WIDTH = 512
    DEFAULT_STEPS = 20
    DEFAULT_GUIDANCE_SCALE = 7.5
    DEFAULT_NUM_SAMPLES = 5
    DEFAULT_BATCH_SIZE = 1
    
    # Default negative prompt for better quality
    DEFAULT_NEGATIVE_PROMPT = (
        "blurry, low quality, distorted, deformed, artifacts, watermark, "
        "text, signature, worst quality, low resolution, pixelated"
    )
    
    # Disease definitions with symptoms
    DISEASE_SYMPTOMS = {
        "Tilapia_Lake_Virus": "swollen abdomen and hemorrhagic fins",
        "Streptococcus": "red hemorrhagic streaks and swollen gill area",
        "Parasitic_Disease": "visible worm-like parasites attached to scales",
        "Ichthyophthirius": "white ich spots and fuzzy patches on its fins",
        "Fungal_Disease": "cotton-like white fungal growth patches on fins and body",
        "Flavobacterium": "yellowish gummy sheen and rotten tail fin",
        "Epizootic_Ulcerative_Syndrome": "large ulcerative lesions with irregular edges",
        "Edwardsiella_Ictaluri": "dark hemorrhagic lesions and skin darkening",
        "Columnaris_Disease": "grayish mouth rot and skin necrosis around head",
        "Aeromonas_Septicemia": "reddened body and hemorrhagic spots across its body",
    }
    
    # View templates for different angles/perspectives
    VIEW_TEMPLATES = [
        "a photorealistic studio shot of a tilapia fish on a clean white seamless backdrop, side profile, full fish visible",
        "a photorealistic studio macro shot of a tilapia fish on a clean white seamless backdrop, close-up on the lesion",
    ]
    
    # Supported image formats
    SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg']


class ModelManager:
    """Manages model loading and optimization"""
    
    def __init__(self):
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Enable optimizations
        self._setup_optimizations()
    
    def _setup_optimizations(self) -> None:
        """Setup PyTorch optimizations"""
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
    
    def load_model_with_lora(self, model_name: str, lora_weights: str) -> None:
        """Load base model and apply LoRA weights"""
        print(f"Loading base model: {model_name}")
        
        # Load pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True
        )
        
        # Use better scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="dpmsolver++"
        )
        
        # Move to device
        self.pipe = self.pipe.to(self.device)
        
        # Load LoRA weights
        print(f"Loading LoRA weights from: {lora_weights}")
        self.pipe.unet = PeftModel.from_pretrained(self.pipe.unet, lora_weights)
        
        # Apply memory optimizations
        self._apply_memory_optimizations()
    
    def _apply_memory_optimizations(self) -> None:
        """Apply memory and performance optimizations"""
        self.pipe.enable_attention_slicing()
        self.pipe.enable_model_cpu_offload()
        
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("xformers memory efficient attention enabled")
        except Exception as e:
            print(f"xformers not available: {e}")
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.pipe is not None


class ImageGenerator:
    """Handles image generation with batch processing"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.pipe = model_manager.pipe
    
    def generate_batch(
        self,
        prompts: List[str],
        negative_prompt: str,
        generator: Optional[torch.Generator],
        generation_params: Dict
    ) -> List[Image.Image]:
        """Generate images in batches with memory management"""
        all_images = []
        batch_size = generation_params.get('batch_size', 1)
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_neg_prompts = [negative_prompt] * len(batch_prompts) if negative_prompt else None
            
            try:
                images = self._generate_batch_attempt(
                    batch_prompts, batch_neg_prompts, generator, generation_params
                )
                all_images.extend(images)
                
                # Clear cache after each batch
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"GPU memory error in batch {i//batch_size + 1}, trying individual generation")
                    images = self._generate_individual_fallback(
                        batch_prompts, negative_prompt, generator, generation_params
                    )
                    all_images.extend(images)
                else:
                    raise e
        
        return all_images
    
    def _generate_batch_attempt(
        self,
        prompts: List[str],
        negative_prompts: Optional[List[str]],
        generator: Optional[torch.Generator],
        params: Dict
    ) -> List[Image.Image]:
        """Attempt batch generation"""
        outputs = self.pipe(
            prompts,
            negative_prompt=negative_prompts,
            generator=generator,
            height=params['height'],
            width=params['width'],
            num_inference_steps=params['steps'],
            guidance_scale=params['guidance_scale'],
            num_images_per_prompt=1
        )
        return outputs.images
    
    def _generate_individual_fallback(
        self,
        prompts: List[str],
        negative_prompt: str,
        generator: Optional[torch.Generator],
        params: Dict
    ) -> List[Image.Image]:
        """Fallback to individual image generation"""
        images = []
        
        for prompt in prompts:
            try:
                output = self.pipe(
                    [prompt],
                    negative_prompt=[negative_prompt] if negative_prompt else None,
                    generator=generator,
                    height=params['height'],
                    width=params['width'],
                    num_inference_steps=params['steps'],
                    guidance_scale=params['guidance_scale']
                )
                images.extend(output.images)
                torch.cuda.empty_cache()
            except Exception as inner_e:
                print(f"Failed to generate image for prompt: {prompt[:50]}...")
                print(f"Error: {inner_e}")
        
        return images


class InferenceManager:
    """Main class for managing the inference process"""
    
    def __init__(self, config: DiffusionInferenceConfig):
        self.config = config
        self.model_manager = ModelManager()
        self.image_generator = None
        self.generation_stats = {
            'total_generated': 0,
            'total_time': 0,
            'memory_usage': []
        }
    
    def setup_model(self, model_name: str, lora_weights: str) -> None:
        """Setup the model for inference"""
        self.model_manager.load_model_with_lora(model_name, lora_weights)
        self.image_generator = ImageGenerator(self.model_manager)
    
    def generate_disease_dataset(
        self,
        output_dir: str,
        generation_params: Dict,
        diseases: Optional[List[str]] = None
    ) -> Dict:
        """Generate a complete dataset for all diseases"""
        if diseases is None:
            diseases = list(self.config.DISEASE_SYMPTOMS.keys())
        
        # Setup output directory
        output_base = self._setup_output_directory(output_dir)
        
        # Setup generator for reproducibility
        generator = self._setup_generator(generation_params.get('seed'))
        
        # Save generation configuration
        self._save_generation_config(output_base, generation_params, diseases)
        
        print(f"Generating {len(diseases)} diseases × {len(self.config.VIEW_TEMPLATES)} views × {generation_params['num_samples']} samples")
        print(f"Total images: {len(diseases) * len(self.config.VIEW_TEMPLATES) * generation_params['num_samples']}")
        print(f"Output directory: {output_base}")
        
        # Generate images for each disease
        results = {}
        
        for disease_idx, disease in enumerate(diseases):
            print(f"\n📊 Processing {disease} ({disease_idx + 1}/{len(diseases)})")
            
            disease_results = self._generate_disease_images(
                disease, output_base, generation_params, generator
            )
            results[disease] = disease_results
            
            # Update statistics
            self.generation_stats['total_generated'] += disease_results['images_generated']
        
        # Save final results
        self._save_generation_results(output_base, results)
        
        return results
    
    def _setup_output_directory(self, base_dir: str) -> Path:
        """Setup timestamped output directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(base_dir) / f"inference_{timestamp}"
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path
    
    def _setup_generator(self, seed: Optional[int]) -> Optional[torch.Generator]:
        """Setup random generator for reproducibility"""
        if seed is not None:
            generator = torch.Generator(device=self.model_manager.device).manual_seed(seed)
            print(f"Using seed: {seed}")
            return generator
        return None
    
    def _generate_disease_images(
        self,
        disease: str,
        output_base: Path,
        generation_params: Dict,
        generator: Optional[torch.Generator]
    ) -> Dict:
        """Generate images for a specific disease"""
        # Create disease directory
        disease_dir = output_base / disease
        disease_dir.mkdir(exist_ok=True)
        
        disease_stats = {
            'disease': disease,
            'images_generated': 0,
            'views': []
        }
        
        # Get disease symptoms
        symptoms = self.config.DISEASE_SYMPTOMS.get(disease, "unknown disease symptoms")
        
        for view_idx, view_template in enumerate(self.config.VIEW_TEMPLATES):
            print(f"View {view_idx + 1}/{len(self.config.VIEW_TEMPLATES)}: {view_template[:50]}...")
            
            # Create prompts for this view
            base_prompt = f"{view_template}, showing {symptoms}, high-detail scientific imaging style, soft even lighting"
            prompts = [base_prompt] * generation_params['num_samples']
            
            # Generate images
            images = self.image_generator.generate_batch(
                prompts,
                generation_params['negative_prompt'],
                generator,
                generation_params
            )
            
            # Save images
            view_stats = self._save_disease_view_images(
                images, disease_dir, disease, view_idx
            )
            
            disease_stats['views'].append(view_stats)
            disease_stats['images_generated'] += view_stats['count']
            
            # Clear memory
            torch.cuda.empty_cache()
        
        return disease_stats
    
    def _save_disease_view_images(
        self,
        images: List[Image.Image],
        disease_dir: Path,
        disease: str,
        view_idx: int
    ) -> Dict:
        """Save images for a specific disease and view"""
        saved_count = 0
        
        for img_idx, img in enumerate(images):
            if img is not None:
                filename = f"{disease}_view{view_idx}_sample{img_idx:02d}.png"
                filepath = disease_dir / filename
                img.save(filepath)
                saved_count += 1
                print(f"Saved: {filename}")
        
        return {
            'view_index': view_idx,
            'count': saved_count,
            'total_requested': len(images)
        }
    
    def _save_generation_config(
        self,
        output_base: Path,
        generation_params: Dict,
        diseases: List[str]
    ) -> None:
        """Save generation configuration"""
        config_data = {
            "model_name": generation_params.get('model_name', 'unknown'),
            "lora_weights": generation_params.get('lora_weights', 'unknown'),
            "parameters": {
                key: val for key, val in generation_params.items()
                if key not in ['model_name', 'lora_weights']
            },
            "timestamp": datetime.now().isoformat(),
            "diseases": diseases,
            "view_templates": self.config.VIEW_TEMPLATES
        }
        
        with open(output_base / "generation_config.json", "w") as f:
            json.dump(config_data, f, indent=2)
    
    def _save_generation_results(self, output_base: Path, results: Dict) -> None:
        """Save generation results summary"""
        summary = {
            "total_diseases": len(results),
            "total_images_generated": self.generation_stats['total_generated'],
            "gpu_memory_final": f"{torch.cuda.memory_allocated() / 1e9:.1f} GB" if torch.cuda.is_available() else "N/A",
            "results_by_disease": results,
            "generation_timestamp": datetime.now().isoformat()
        }
        
        with open(output_base / "generation_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nGeneration complete!")
        print(f"Total images generated: {self.generation_stats['total_generated']}")
        print(f"Results saved in: {output_base}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Batch inference with a LoRA-fine-tuned Stable Diffusion model"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name", type=str,
        default=DiffusionInferenceConfig.DEFAULT_MODEL_NAME,
        help="Base Stable Diffusion model identifier or path"
    )
    parser.add_argument(
        "--lora_weights", type=str, required=True,
        help="Path to directory with LoRA weights"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir", type=str, default="outputs/inference",
        help="Base directory to save all generated images"
    )
    parser.add_argument(
        "--num_samples", type=int, default=DiffusionInferenceConfig.DEFAULT_NUM_SAMPLES,
        help="Images to generate per prompt"
    )
    
    # Generation arguments
    parser.add_argument(
        "--height", type=int, default=DiffusionInferenceConfig.DEFAULT_HEIGHT,
        help="Output image height"
    )
    parser.add_argument(
        "--width", type=int, default=DiffusionInferenceConfig.DEFAULT_WIDTH,
        help="Output image width"
    )
    parser.add_argument(
        "--steps", type=int, default=DiffusionInferenceConfig.DEFAULT_STEPS,
        help="Number of inference steps (higher = more detail)"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=DiffusionInferenceConfig.DEFAULT_GUIDANCE_SCALE,
        help="Guidance scale (7.5-8.0 recommended)"
    )
    parser.add_argument(
        "--negative_prompt", type=str,
        default=DiffusionInferenceConfig.DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt to improve quality"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--batch_size", type=int, default=DiffusionInferenceConfig.DEFAULT_BATCH_SIZE,
        help="Batch size for inference (reduce if memory issues)"
    )
    
    # Disease selection
    parser.add_argument(
        "--diseases", type=str, nargs='+', default=None,
        help="Specific diseases to generate (default: all)"
    )
    
    return parser.parse_args()


def main():
    """Main inference function"""
    args = parse_arguments()
    
    # Initialize config and manager
    config = DiffusionInferenceConfig()
    inference_manager = InferenceManager(config)
    
    # Setup model
    inference_manager.setup_model(args.model_name, args.lora_weights)
    
    # Prepare generation parameters
    generation_params = {
        'model_name': args.model_name,
        'lora_weights': args.lora_weights,
        'height': args.height,
        'width': args.width,
        'steps': args.steps,
        'guidance_scale': args.guidance_scale,
        'negative_prompt': args.negative_prompt,
        'seed': args.seed,
        'num_samples': args.num_samples,
        'batch_size': args.batch_size
    }
    
    # Generate images
    results = inference_manager.generate_disease_dataset(
        args.output_dir,
        generation_params,
        args.diseases
    )
    
    print(f"\n🎉 Generation completed successfully!")
    for disease, stats in results.items():
        print(f"  {disease}: {stats['images_generated']} images")


if __name__ == "__main__":
    main()
