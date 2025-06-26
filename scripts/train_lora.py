"""
Command-line script for training LoRA models
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.diffusion.lora_trainer import main

if __name__ == "__main__":
    main()
