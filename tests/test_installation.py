"""
Simple test to verify the installation and basic functionality
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all main modules can be imported"""
    try:
        from config.settings import Config
        from src.utils.helpers import parse_range, make_transform_matrix
        from src.generators.model_manager import ModelManager
        from src.generators.image_generator import ImageGenerator
        from src.ui.components import UIComponents
        from src.analysis.metrics_analyzer import MetricsAnalyzer
        
        print("✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without models"""
    try:
        # Test config
        config = Config()
        assert config.PAGE_TITLE == "StyleGAN Image Generator"
        
        # Test helpers
        seeds = parse_range("0,1,2")
        assert seeds == [0, 1, 2]
        
        seeds = parse_range("0-3")
        assert seeds == [0, 1, 2, 3]
        
        # Test transform matrix
        matrix = make_transform_matrix((0.5, 0.5), 45)
        assert matrix.shape == (3, 3)
        
        # Test analyzer
        analyzer = MetricsAnalyzer()
        smoothed = analyzer.calculate_moving_average([1, 2, 3, 4, 5], 3)
        assert len(smoothed) == 5
        
        print("✓ Basic functionality tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Functionality test error: {e}")
        return False

def test_directory_structure():
    """Test that required directories exist"""
    required_dirs = [
        "src",
        "config", 
        "data/models",
        "data/metrics",
        "outputs",
        "scripts"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✓ {dir_path} exists")
        else:
            print(f"✗ {dir_path} missing")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    print("GAN Image Generator - Installation Test")
    print("=" * 40)
    
    print("\n1. Testing imports...")
    imports_ok = test_imports()
    
    print("\n2. Testing basic functionality...")
    functionality_ok = test_basic_functionality()
    
    print("\n3. Testing directory structure...")
    structure_ok = test_directory_structure()
    
    print("\n" + "=" * 40)
    if all([imports_ok, functionality_ok, structure_ok]):
        print("✓ All tests passed! Installation is successful.")
        print("\nYou can now run:")
        print("  streamlit run app.py")
        return 0
    else:
        print("✗ Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
