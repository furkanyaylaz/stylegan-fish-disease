"""
FID Score Visualization Script
Clean implementation for visualizing FID progression
"""
import sys
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.metrics_analyzer import MetricsAnalyzer


def main():
    """Generate FID visualization plots"""
    parser = argparse.ArgumentParser(description="Generate FID score visualization")
    parser.add_argument(
        "--metrics-dir",
        default="data/metrics",
        help="Directory containing metrics files"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/plots",
        help="Directory to save plots"
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Apply smoothing to the plots"
    )
    
    args = parser.parse_args()
    
    metrics_dir = Path(args.metrics_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find FID files
    fid_files = list(metrics_dir.glob("*fid*.jsonl"))
    
    if not fid_files:
        print(f"No FID files found in {metrics_dir}")
        return
    
    print(f"Found {len(fid_files)} FID files")
    
    # Initialize analyzer
    analyzer = MetricsAnalyzer()
    
    # Prepare data
    file_paths = [str(f) for f in fid_files]
    model_names = []
    
    for file_path in fid_files:
        name = file_path.name.replace('.jsonl', '')
        name = name.replace('_fid50k', '').replace('_metric-fid50k_full', '')
        model_names.append(name)
    
    # Generate plot
    print("Generating FID comparison plot...")
    fig = analyzer.plot_fid_progression(
        file_paths,
        model_names,
        str(output_dir / "fid_comparison.png"),
        smooth=args.smooth
    )
    
    print(f"Plot saved to {output_dir / 'fid_comparison.png'}")


if __name__ == "__main__":
    main()
