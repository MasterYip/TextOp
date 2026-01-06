"""
Quick test script to verify dyed data visualization setup.
"""

import sys
from pathlib import Path

# Add paths
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from replay_buffer import ReplayBuffer
import numpy as np


def test_replay_buffer():
    """Test that ReplayBuffer can be imported and used."""
    print("Testing ReplayBuffer import... ", end="")
    try:
        buffer = ReplayBuffer.create_empty_numpy()
        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False


def test_motionclip_import():
    """Test that MotionCLIP can be imported as package."""
    print("Testing MotionCLIP package import... ", end="")
    try:
        import motionclip
        from motionclip import get_motion_clip, get_motion_text_mapping
        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False


def test_visualize_utils():
    """Test that visualization utilities work."""
    print("Testing visualization utilities... ", end="")
    try:
        from visualize_utils import G1MotionVisualizer
        viz = G1MotionVisualizer()
        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False


def test_dyed_data_exists():
    """Check if sample dyed data exists."""
    print("Checking for dyed data... ", end="")
    
    # Check common locations
    possible_paths = [
        Path("../../../artifacts/g1_multimotion_noise_median/motion_dyed.zarr"),
        Path("../../../artifacts/g1_multimotion_noise/motion_dyed.zarr"),
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"✓ Found at {path}")
            return True
    
    print("✗ No dyed data found (this is OK if you haven't run data_dyeing.py yet)")
    return False


def test_checkpoint_exists():
    """Check if MotionCLIP checkpoint exists."""
    print("Checking for MotionCLIP checkpoint... ", end="")
    
    # Check common locations
    possible_paths = [
        Path("../../../../MotionCLIP/exps/g1-model-xyz/checkpoint_0100.pth.tar"),
        Path("../../../../MotionCLIP/exps/g1-model-rot6d/checkpoint_0100.pth.tar"),
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"✓ Found at {path}")
            return True
    
    print("✗ No checkpoint found")
    return False


def main():
    print("="*80)
    print("DYED DATA VISUALIZATION - SETUP CHECK")
    print("="*80)
    print()
    
    results = []
    
    # Run tests
    results.append(("ReplayBuffer", test_replay_buffer()))
    results.append(("MotionCLIP Package", test_motionclip_import()))
    results.append(("Visualization Utils", test_visualize_utils()))
    results.append(("Dyed Data", test_dyed_data_exists()))
    results.append(("Checkpoint", test_checkpoint_exists()))
    
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:30s} {status}")
    
    print()
    print(f"Total: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n✓ All checks passed! Ready to run dyed_data_vis.py")
    elif passed >= total - 2:
        print("\n⚠ Most checks passed. You may need to run data_dyeing.py first.")
    else:
        print("\n✗ Some checks failed. Please install missing dependencies.")
    
    print("="*80)


if __name__ == '__main__':
    main()
