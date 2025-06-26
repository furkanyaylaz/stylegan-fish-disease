"""
Universal runner script for the GAN Image Generator project
Provides easy access to all major functionalities
"""
import argparse
import subprocess
import sys
from pathlib import Path


def run_stylegan_app():
    """Run the main StyleGAN Streamlit application"""
    print("Starting StyleGAN Image Generator...")
    subprocess.run([sys.executable, "app.py"])


def run_analysis():
    """Run comprehensive GAN training analysis"""
    print("Running GAN training analysis...")
    subprocess.run([sys.executable, "scripts/comprehensive_analysis.py"])


def run_fid_visualization():
    """Run FID score visualization"""
    print("Generating FID visualizations...")
    subprocess.run([sys.executable, "scripts/fid_visualization.py", "--smooth"])


def run_train_lora(args):
    """Run LoRA training with provided arguments"""
    print("Starting LoRA training...")
    cmd = [sys.executable, "scripts/train_lora.py"]
    
    if args.data_dir:
        cmd.extend(["--data_dir", args.data_dir])
    if args.output_dir:
        cmd.extend(["--output_dir", args.output_dir])
    if args.epochs:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.batch_size:
        cmd.extend(["--batch_size", str(args.batch_size)])
    if args.learning_rate:
        cmd.extend(["--learning_rate", str(args.learning_rate)])
    
    subprocess.run(cmd)


def run_generate_images(args):
    """Run image generation with LoRA model"""
    print("Generating images with LoRA model...")
    cmd = [sys.executable, "scripts/generate_images.py"]
    
    if args.lora_weights:
        cmd.extend(["--lora_weights", args.lora_weights])
    if args.output_dir:
        cmd.extend(["--output_dir", args.output_dir])
    if args.num_samples:
        cmd.extend(["--num_samples", str(args.num_samples)])
    
    subprocess.run(cmd)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="GAN Image Generator - Universal Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py app                                    # Run StyleGAN web app
  python run.py analysis                               # Run training analysis
  python run.py fid                                    # Generate FID plots
  python run.py train --data_dir data --epochs 20     # Train LoRA model
  python run.py generate --lora_weights model         # Generate images
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # StyleGAN app
    subparsers.add_parser('app', help='Run StyleGAN Streamlit application')
    
    # Analysis
    subparsers.add_parser('analysis', help='Run comprehensive GAN training analysis')
    subparsers.add_parser('fid', help='Generate FID score visualizations')
    
    # LoRA training
    train_parser = subparsers.add_parser('train', help='Train LoRA model')
    train_parser.add_argument('--data_dir', type=str, help='Dataset directory')
    train_parser.add_argument('--output_dir', type=str, help='Output directory')
    train_parser.add_argument('--epochs', type=int, help='Number of epochs')
    train_parser.add_argument('--batch_size', type=int, help='Batch size')
    train_parser.add_argument('--learning_rate', type=float, help='Learning rate')
    
    # Image generation
    gen_parser = subparsers.add_parser('generate', help='Generate images with LoRA')
    gen_parser.add_argument('--lora_weights', type=str, help='LoRA weights path')
    gen_parser.add_argument('--output_dir', type=str, help='Output directory')
    gen_parser.add_argument('--num_samples', type=int, help='Number of samples')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Change to project directory
    project_root = Path(__file__).parent
    import os
    os.chdir(project_root)
    
    # Route to appropriate function
    if args.command == 'app':
        run_stylegan_app()
    elif args.command == 'analysis':
        run_analysis()
    elif args.command == 'fid':
        run_fid_visualization()
    elif args.command == 'train':
        run_train_lora(args)
    elif args.command == 'generate':
        run_generate_images(args)
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()


if __name__ == "__main__":
    main()
