"""
LoRA Fine-tuning Manager for Stable Diffusion Models
Clean implementation with proper error handling and configuration management
"""
import os
import sys
import argparse
import json
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import get_peft_model, LoraConfig
from transformers import CLIPTextModel

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Config


class DiffusionConfig:
    """Configuration class for diffusion model training"""
    
    # Model settings
    DEFAULT_MODEL_NAME = "runwayml/stable-diffusion-v1-5"
    DEFAULT_RESOLUTION = 512
    DEFAULT_BATCH_SIZE = 2
    DEFAULT_LEARNING_RATE = 1e-4
    DEFAULT_EPOCHS = 20
    
    # LoRA settings
    DEFAULT_LORA_RANK = 32
    DEFAULT_LORA_ALPHA = 32
    DEFAULT_LORA_DROPOUT = 0.1
    
    # Training settings
    DEFAULT_GRADIENT_ACCUMULATION_STEPS = 6
    DEFAULT_VALIDATION_SPLIT = 0.1
    DEFAULT_WARMUP_STEPS = 100
    DEFAULT_SAVE_EVERY = 5
    DEFAULT_SEED = 42
    
    # Target modules for LoRA
    LORA_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]
    
    # Disease prompt templates
    DISEASE_PROMPTS = {
        'ich': [
            "fish with ich disease, white spot disease, parasitic infection, aquarium fish, detailed clinical photo",
            "tropical fish suffering from ich, white spots on skin and fins, fish pathology, veterinary documentation",
            "sick fish with white spot disease, parasitic skin infection, aquatic veterinary medicine",
            "fish disease ich, white pustules on body, marine biology, underwater photography"
        ],
        'fin_rot': [
            "fish with fin rot disease, damaged fins, bacterial infection, fish pathology, clinical documentation",
            "diseased fish showing fin rot symptoms, deteriorating fins, aquatic disease, veterinary photo",
            "bacterial fin rot in fish, rotting fin edges, fish health documentation, medical photography",
            "sick fish with fin damage, bacterial infection, aquarium disease, clinical study"
        ],
        'fungal': [
            "fish with fungal infection, cotton-like growth, aquatic fungal disease, veterinary documentation",
            "diseased fish with fungal patches, white fluffy growth, fish pathology, clinical photo",
            "fungal infection in fish, cotton mouth disease, aquatic medicine, underwater photography",
            "sick fish with fungal growth, white cottony patches, fish disease documentation"
        ],
        'bacterial': [
            "fish with bacterial infection, skin lesions, fish pathology, veterinary documentation",
            "diseased fish showing bacterial symptoms, red inflamed areas, aquatic disease, clinical photo",
            "bacterial infection in fish, skin ulcers, fish health documentation, medical photography",
            "sick fish with bacterial disease, infected wounds, aquarium pathology, clinical study"
        ],
        'tilapia_lake_virus': [
            "tilapia with lake virus infection, viral disease, fish pathology, aquaculture veterinary documentation",
            "diseased tilapia showing viral symptoms, TiLV infection, fish health documentation, clinical photo",
            "tilapia lake virus disease, viral infection in fish, aquatic veterinary medicine, underwater photography",
            "sick tilapia with viral disease, TiLV pathology, fish disease documentation, medical photography"
        ],
        'streptococcus': [
            "fish with streptococcus infection, bacterial disease, fish pathology, veterinary documentation",
            "diseased fish showing streptococcal symptoms, bacterial infection, aquatic disease, clinical photo",
            "streptococcus infection in fish, bacterial pathology, fish health documentation, medical photography",
            "sick fish with streptococcal disease, bacterial infection, aquarium pathology, clinical study"
        ],
        'parasitic_disease': [
            "fish with parasitic disease, external parasites, fish pathology, veterinary documentation",
            "diseased fish showing parasitic symptoms, parasite infection, aquatic disease, clinical photo",
            "parasitic infection in fish, external parasites, fish health documentation, medical photography",
            "sick fish with parasite disease, parasitic infection, aquarium pathology, clinical study"
        ],
        'ichthyophthirius': [
            "fish with ichthyophthirius infection, white spot disease, parasitic infection, detailed clinical photo",
            "diseased fish with ichthyophthirius parasites, white spots, fish pathology, veterinary documentation",
            "ichthyophthirius disease in fish, parasitic skin infection, aquatic veterinary medicine",
            "sick fish with ichthyophthirius, white pustules on body, fish disease documentation"
        ],
        'fungal_disease': [
            "fish with fungal disease, cotton-like growth, aquatic fungal infection, veterinary documentation",
            "diseased fish with fungal pathology, white fluffy growth, fish health documentation, clinical photo",
            "fungal disease in fish, cotton mouth infection, aquatic medicine, underwater photography",
            "sick fish with fungal pathology, white cottony patches, fish disease documentation"
        ],
        'flavobacterium': [
            "fish with flavobacterium infection, bacterial disease, fish pathology, veterinary documentation",
            "diseased fish showing flavobacterium symptoms, bacterial infection, aquatic disease, clinical photo",
            "flavobacterium infection in fish, bacterial pathology, fish health documentation, medical photography",
            "sick fish with flavobacterium disease, bacterial infection, aquarium pathology, clinical study"
        ],
        'epizootic_ulcerative_syndrome': [
            "fish with epizootic ulcerative syndrome, skin ulcers, fish pathology, veterinary documentation",
            "diseased fish showing EUS symptoms, ulcerative lesions, aquatic disease, clinical photo",
            "epizootic ulcerative syndrome in fish, skin ulcers, fish health documentation, medical photography",
            "sick fish with EUS disease, ulcerative infection, aquarium pathology, clinical study"
        ],
        'edwardsiella_ictaluri': [
            "fish with edwardsiella ictaluri infection, bacterial disease, fish pathology, veterinary documentation",
            "diseased fish showing edwardsiella symptoms, bacterial infection, aquatic disease, clinical photo",
            "edwardsiella ictaluri infection in fish, bacterial pathology, fish health documentation, medical photography",
            "sick fish with edwardsiella disease, bacterial infection, aquarium pathology, clinical study"
        ],
        'columnaris_disease': [
            "fish with columnaris disease, bacterial infection, fish pathology, veterinary documentation",
            "diseased fish showing columnaris symptoms, bacterial infection, aquatic disease, clinical photo",
            "columnaris disease in fish, bacterial pathology, fish health documentation, medical photography",
            "sick fish with columnaris infection, bacterial disease, aquarium pathology, clinical study"
        ],
        'aeromonas_septicemia': [
            "fish with aeromonas septicemia, bacterial infection, fish pathology, veterinary documentation",
            "diseased fish showing aeromonas symptoms, septicemia, aquatic disease, clinical photo",
            "aeromonas septicemia in fish, bacterial pathology, fish health documentation, medical photography",
            "sick fish with aeromonas infection, septicemia disease, aquarium pathology, clinical study"
        ]
    }
    
    DEFAULT_PROMPTS = [
        "diseased fish, fish pathology, aquatic disease, veterinary documentation, clinical photography",
        "sick fish, fish health problems, aquatic medicine, underwater disease documentation",
        "fish disease symptoms, aquatic pathology, veterinary medicine, clinical fish photography",
        "unhealthy fish, fish illness, aquatic veterinary science, disease documentation"
    ]


class FishDiseaseDataset(Dataset):
    """Dataset class for fish disease images with enhanced prompt generation"""
    
    def __init__(
        self,
        root_dir: str,
        feature_extractor,
        tokenizer,
        resolution: int = 512,
        validation_split: float = 0.1,
        is_validation: bool = False,
        seed: int = 42
    ):
        self.root_dir = Path(root_dir)
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.resolution = resolution
        
        # Load all samples
        all_samples = self._load_samples()
        
        # Split data
        self.samples = self._split_data(all_samples, validation_split, is_validation, seed)
        
        print(f"{'Validation' if is_validation else 'Training'} dataset loaded: {len(self.samples)} samples")
    
    def _load_samples(self) -> List[Dict]:
        """Load all samples from the dataset directory"""
        all_samples = []
        
        for label_dir in sorted(self.root_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            
            label = label_dir.name
            disease_key = label.lower().replace('_', '').replace('-', '')
            prompts = DiffusionConfig.DISEASE_PROMPTS.get(disease_key, DiffusionConfig.DEFAULT_PROMPTS)
            
            for image_path in label_dir.glob('*'):
                if image_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    prompt = prompts[len(all_samples) % len(prompts)]
                    all_samples.append({
                        "path": str(image_path),
                        "prompt": prompt,
                        "label": label
                    })
        
        return all_samples
    
    def _split_data(
        self,
        all_samples: List[Dict],
        validation_split: float,
        is_validation: bool,
        seed: int
    ) -> List[Dict]:
        """Split data into training and validation sets"""
        np.random.seed(seed)
        indices = np.random.permutation(len(all_samples))
        val_size = int(len(all_samples) * validation_split)
        
        if is_validation:
            selected_indices = indices[:val_size]
        else:
            selected_indices = indices[val_size:]
        
        return [all_samples[i] for i in selected_indices]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample"""
        item = self.samples[idx]
        
        # Load and preprocess image
        image = self._load_image(item["path"])
        
        # Process image
        pixel_values = self.feature_extractor(
            images=image,
            return_tensors="pt",
            do_normalize=True,
            do_resize=False
        ).pixel_values[0]
        
        # Tokenize prompt
        input_ids = self.tokenizer(
            item["prompt"],
            truncation=True,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        ).input_ids[0]
        
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids
        }
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load and preprocess image with error handling"""
        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return black image as fallback
            return Image.new("RGB", (self.resolution, self.resolution), color="black")


class LoRATrainer:
    """Main trainer class for LoRA fine-tuning"""
    
    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        self.optimizer = None
        self.scheduler = None
        self.training_history = []
        self.best_val_loss = float('inf')
        
    def setup_model(self, model_name: str, lora_config: Dict) -> None:
        """Setup the diffusion model with LoRA"""
        print(f"Loading Stable Diffusion pipeline: {model_name}")
        
        # Load pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True
        )
        
        # Move to device
        self.pipe.to(self.device)
        
        # Setup LoRA
        lora_config_obj = LoraConfig(
            r=lora_config['rank'],
            lora_alpha=lora_config['alpha'],
            target_modules=self.config.LORA_TARGET_MODULES,
            lora_dropout=lora_config['dropout'],
            bias="none"
        )
        
        self.pipe.unet = get_peft_model(self.pipe.unet, lora_config_obj)
        
        # Apply optimizations
        self._apply_optimizations()
        
        # Print parameter info
        self._print_parameter_info()
        
        # Freeze components
        self._freeze_components()
    
    def _apply_optimizations(self) -> None:
        """Apply memory and performance optimizations"""
        self.pipe.enable_attention_slicing()
        self.pipe.enable_model_cpu_offload()
        
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("xformers memory efficient attention enabled")
        except Exception as e:
            print(f"xformers not available: {e}")
    
    def _print_parameter_info(self) -> None:
        """Print information about trainable parameters"""
        trainable_params = sum(p.numel() for p in self.pipe.unet.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.pipe.unet.parameters())
        
        print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    def _freeze_components(self) -> None:
        """Freeze text encoder and VAE components"""
        for param in self.pipe.text_encoder.parameters():
            param.requires_grad = False
        for param in self.pipe.vae.parameters():
            param.requires_grad = False
    
    def setup_training(self, learning_rate: float, total_steps: int) -> None:
        """Setup optimizer and scheduler"""
        self.optimizer = torch.optim.AdamW(
            self.pipe.unet.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=total_steps // 4,
            T_mult=2,
            eta_min=learning_rate * 0.01
        )
        
        # Set training mode
        self.pipe.unet.train()
        self.pipe.text_encoder.eval()
        self.pipe.vae.eval()
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        gradient_accumulation_steps: int
    ) -> float:
        """Train for one epoch"""
        epoch_loss = 0.0
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            loss = self._process_batch(batch)
            
            # Scale loss and backward
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            epoch_loss += loss.item() * gradient_accumulation_steps
            
            # Update weights
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.pipe.unet.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Print progress
                if (batch_idx + 1) // gradient_accumulation_steps % 10 == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    step = (batch_idx + 1) // gradient_accumulation_steps
                    print(f"Epoch {epoch} Step {step} | Loss: {loss.item() * gradient_accumulation_steps:.4f} | LR: {current_lr:.2e}")
        
        return epoch_loss / len(train_loader)
    
    def _process_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process a single batch"""
        pixel_values = batch["pixel_values"].to(self.device, dtype=torch.float16)
        input_ids = batch["input_ids"].to(self.device)
        
        # Encode images to latent space
        with torch.no_grad():
            latents = self.pipe.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * self.pipe.vae.config.scaling_factor
        
        # Sample noise and timesteps
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.pipe.scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=self.device
        ).long()
        
        # Add noise to latents
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timesteps)
        
        # Get text embeddings
        with torch.no_grad():
            encoder_hidden_states = self.pipe.text_encoder(input_ids)[0]
        
        # Predict noise
        model_pred = self.pipe.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False
        )[0]
        
        # Calculate loss
        return F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
    
    def validate(self, val_loader: DataLoader) -> float:
        """Run validation and return average loss"""
        self.pipe.unet.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                loss = self._process_batch(batch)
                total_loss += loss.item()
                num_batches += 1
        
        self.pipe.unet.train()
        return total_loss / num_batches if num_batches > 0 else float('inf')
    
    def save_checkpoint(
        self,
        output_dir: str,
        epoch: int,
        loss: float,
        is_best: bool = False
    ) -> None:
        """Save training checkpoint"""
        if is_best:
            checkpoint_dir = Path(output_dir) / "best_model"
        else:
            checkpoint_dir = Path(output_dir) / f"checkpoint-epoch-{epoch}"
        
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA weights
        self.pipe.unet.save_pretrained(str(checkpoint_dir))
        
        # Save training state
        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'model_name': getattr(self.pipe.unet.config, '_name_or_path', 'unknown')
        }, checkpoint_dir / 'training_state.pt')
        
        print(f"{'Best model' if is_best else 'Checkpoint'} saved at {checkpoint_dir}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Fine-tune Stable Diffusion with LoRA on fish disease images"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name", type=str,
        default=DiffusionConfig.DEFAULT_MODEL_NAME,
        help="Pretrained Stable Diffusion model identifier or local path"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to root dataset directory with class subfolders"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save LoRA weights"
    )
    
    # Training arguments
    parser.add_argument(
        "--resolution", type=int, default=DiffusionConfig.DEFAULT_RESOLUTION,
        help="Image resolution (height and width)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=DiffusionConfig.DEFAULT_BATCH_SIZE,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=DiffusionConfig.DEFAULT_LEARNING_RATE,
        help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--epochs", type=int, default=DiffusionConfig.DEFAULT_EPOCHS,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=DiffusionConfig.DEFAULT_GRADIENT_ACCUMULATION_STEPS,
        help="Number of steps to accumulate gradients"
    )
    
    # LoRA arguments
    parser.add_argument(
        "--lora_rank", type=int, default=DiffusionConfig.DEFAULT_LORA_RANK,
        help="LoRA rank parameter"
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=DiffusionConfig.DEFAULT_LORA_ALPHA,
        help="LoRA alpha parameter"
    )
    
    # Other arguments
    parser.add_argument(
        "--save_every", type=int, default=DiffusionConfig.DEFAULT_SAVE_EVERY,
        help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--seed", type=int, default=DiffusionConfig.DEFAULT_SEED,
        help="Random seed"
    )
    parser.add_argument(
        "--validation_split", type=float, default=DiffusionConfig.DEFAULT_VALIDATION_SPLIT,
        help="Fraction of data to use for validation"
    )
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_arguments()
    
    # Setup reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    # Print system info
    print(f"Using GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Initialize config and trainer
    config = DiffusionConfig()
    trainer = LoRATrainer(config)
    
    # Setup model
    lora_config = {
        'rank': args.lora_rank,
        'alpha': args.lora_alpha,
        'dropout': config.DEFAULT_LORA_DROPOUT
    }
    trainer.setup_model(args.model_name, lora_config)
    
    # Prepare datasets
    print("Loading datasets...")
    train_dataset = FishDiseaseDataset(
        args.data_dir,
        trainer.pipe.feature_extractor,
        trainer.pipe.tokenizer,
        resolution=args.resolution,
        validation_split=args.validation_split,
        is_validation=False,
        seed=args.seed
    )
    
    val_dataset = FishDiseaseDataset(
        args.data_dir,
        trainer.pipe.feature_extractor,
        trainer.pipe.tokenizer,
        resolution=args.resolution,
        validation_split=args.validation_split,
        is_validation=True,
        seed=args.seed
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )
    
    # Setup training
    total_steps = args.epochs * len(train_loader)
    trainer.setup_training(args.learning_rate, total_steps)
    
    print(f"Starting training for {args.epochs} epochs...")
    print(f"Total training batches per epoch: {len(train_loader)}")
    print(f"Total validation batches: {len(val_loader)}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Train epoch
        avg_train_loss = trainer.train_epoch(train_loader, epoch, args.gradient_accumulation_steps)
        
        # Validate
        val_loss = trainer.validate(val_loader)
        
        # Epoch summary
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch}/{args.epochs} completed in {epoch_time:.1f}s")
        print(f"Average Train Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
        
        # Save training history
        trainer.training_history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'learning_rate': trainer.scheduler.get_last_lr()[0],
            'epoch_time': epoch_time
        })
        
        # Save checkpoints
        is_best = val_loss < trainer.best_val_loss
        if epoch % args.save_every == 0 or is_best:
            trainer.save_checkpoint(args.output_dir, epoch, avg_train_loss, is_best)
            if is_best:
                trainer.best_val_loss = val_loss
                print(f"New best validation loss: {trainer.best_val_loss:.4f}")
        
        # Clear cache
        torch.cuda.empty_cache()
    
    # Final save
    print("\nTraining completed!")
    final_dir = Path(args.output_dir) / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.pipe.unet.save_pretrained(str(final_dir))
    print(f"Final LoRA weights saved at {final_dir}")
    
    # Save training configuration
    config_data = {
        'model_name': args.model_name,
        'resolution': args.resolution,
        'batch_size': args.batch_size,
        'effective_batch_size': args.batch_size * args.gradient_accumulation_steps,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'lora_rank': args.lora_rank,
        'lora_alpha': args.lora_alpha,
        'final_train_loss': avg_train_loss,
        'best_val_loss': trainer.best_val_loss,
        'total_train_samples': len(train_dataset),
        'total_val_samples': len(val_dataset),
        'training_history': trainer.training_history
    }
    
    with open(Path(args.output_dir) / 'training_config.json', 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"Training configuration saved!")
    print(f"Best validation loss achieved: {trainer.best_val_loss:.4f}")
    print(f"Total training time: {sum(h['epoch_time'] for h in trainer.training_history):.1f}s")


if __name__ == "__main__":
    main()
