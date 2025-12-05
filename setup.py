"""
Setup script to install dependencies and verify installation.
"""

import subprocess
import sys

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
    ])
    print("\nInstallation complete!")

def verify_installation():
    """Verify that all packages are installed correctly."""
    print("\nVerifying installation...")
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError:
        print("✗ NumPy not found")
        return False
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
    except ImportError:
        print("✗ PyTorch not found")
        return False
    
    try:
        import scipy
        print(f"✓ SciPy {scipy.__version__}")
    except ImportError:
        print("✗ SciPy not found")
        return False
    
    try:
        import pandas as pd
        print(f"✓ Pandas {pd.__version__}")
    except ImportError:
        print("✗ Pandas not found")
        return False
    
    try:
        import sklearn
        print(f"✓ Scikit-learn {sklearn.__version__}")
    except ImportError:
        print("✗ Scikit-learn not found")
        return False
    
    try:
        from tqdm import tqdm
        print(f"✓ tqdm installed")
    except ImportError:
        print("✗ tqdm not found")
        return False
    
    print("\n✓ All packages installed successfully!")
    return True

def test_dataset_loading():
    """Test loading the CiteSeer dataset."""
    print("\nTesting dataset loading...")
    
    try:
        from data_loader import CiteSeerDataset
        from config import Config
        
        config = Config()
        dataset = CiteSeerDataset(config.DATASET_PATH)
        
        print(f"✓ Dataset loaded successfully!")
        print(f"  Nodes: {dataset.num_nodes}")
        print(f"  Features: {dataset.num_features}")
        print(f"  Classes: {dataset.num_classes}")
        print(f"  Edges: {int(dataset.adj_matrix.sum() / 2)}")
        
        return True
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup KENN replication environment")
    parser.add_argument('--install', action='store_true', help='Install requirements')
    parser.add_argument('--verify', action='store_true', help='Verify installation')
    parser.add_argument('--test', action='store_true', help='Test dataset loading')
    
    args = parser.parse_args()
    
    if not any([args.install, args.verify, args.test]):
        # Run all by default
        args.install = True
        args.verify = True
        args.test = True
    
    if args.install:
        install_requirements()
    
    if args.verify:
        if not verify_installation():
            sys.exit(1)
    
    if args.test:
        if not test_dataset_loading():
            sys.exit(1)
    
    print("\n" + "="*80)
    print("Setup complete! You can now run experiments with:")
    print("  python main.py --quick-test")
    print("="*80)
