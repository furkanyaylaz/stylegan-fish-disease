"""
Run comprehensive analysis and generate reports
"""
import subprocess
import sys
from pathlib import Path


def run_analysis():
    """Run all analysis scripts"""
    project_root = Path(__file__).parent.parent
    
    print("Running comprehensive GAN analysis...")
    
    # Run comprehensive analysis
    print("1. Running comprehensive analysis...")
    subprocess.run([
        sys.executable, 
        str(project_root / "scripts" / "comprehensive_analysis.py")
    ])
    
    # Run FID visualization
    print("2. Generating FID visualizations...")
    subprocess.run([
        sys.executable,
        str(project_root / "scripts" / "fid_visualization.py"),
        "--smooth"
    ])
    
    print("Analysis complete! Check the outputs/ directory for results.")


if __name__ == "__main__":
    run_analysis()
