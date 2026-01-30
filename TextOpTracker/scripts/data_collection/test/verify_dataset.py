"""
Verify collected G1 dataset structure and compatibility with g1_offline_dataset.py

Usage:
    python verify_dataset.py <path_to_zarr_file>
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def verify_dataset(zarr_path: str):
    """Verify that the dataset has the correct structure."""
    
    # Add paths for imports
    ROOT_DIR = str(Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)

    from replay_buffer import ReplayBuffer
    
    print(f"Loading dataset from: {zarr_path}")
    
    # Load dataset
    try:
        buffer = ReplayBuffer.copy_from_path(zarr_path, backend='numpy')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False
    
    print("\n" + "="*60)
    print("Dataset Structure Verification")
    print("="*60)
    
    # Check basic properties
    print(f"\nBasic Properties:")
    print(f"  Number of episodes: {buffer.n_episodes}")
    print(f"  Total timesteps: {buffer.n_steps}")
    print(f"  Episode lengths: min={buffer.episode_lengths.min()}, "
          f"max={buffer.episode_lengths.max()}, mean={buffer.episode_lengths.mean():.1f}")
    
    # Required fields according to OfflineDataset
    required_fields = {
        "act": (29,),  # [T, 29]
        "body_ang_vel": (30, 3),  # [T, 30, 3]
        "body_lin_vel": (30, 3),  # [T, 30, 3]
        "body_pos": (30, 3),  # [T, 30, 3]
        "body_rot": (30, 4),  # [T, 30, 4]
        "joint_pos": (29,),  # [T, 29]
        "joint_vel": (29,),  # [T, 29]
        "root_pos": (3,),  # [T, 3]
        "root_rot": (4,),  # [T, 4]
    }
    
    print(f"\nField Verification:")
    all_fields_valid = True
    
    for field_name, expected_shape in required_fields.items():
        if field_name in buffer.data:
            data = buffer.data[field_name]
            actual_shape = data.shape[1:]  # Skip time dimension
            
            if actual_shape == expected_shape:
                print(f"  ✓ {field_name:15s}: {data.shape} (dtype: {data.dtype})")
            else:
                print(f"  ✗ {field_name:15s}: {data.shape} - Expected shape (T, {expected_shape})")
                all_fields_valid = False
        else:
            print(f"  ✗ {field_name:15s}: MISSING")
            all_fields_valid = False
    
    # Check for unexpected fields
    unexpected_fields = set(buffer.data.keys()) - set(required_fields.keys())
    if unexpected_fields:
        print(f"\nWarning: Unexpected fields found: {unexpected_fields}")
    
    # Data quality checks
    print(f"\nData Quality Checks:")
    
    # Check for NaN or Inf values
    has_invalid_data = False
    for field_name in required_fields.keys():
        if field_name in buffer.data:
            data = buffer.data[field_name]
            if isinstance(data, np.ndarray):
                nan_count = np.isnan(data).sum()
                inf_count = np.isinf(data).sum()
                
                if nan_count > 0 or inf_count > 0:
                    print(f"  ✗ {field_name}: Contains {nan_count} NaN and {inf_count} Inf values")
                    has_invalid_data = True
                else:
                    print(f"  ✓ {field_name}: No invalid values")
    
    # Check quaternion normalization
    if "body_rot" in buffer.data and "root_rot" in buffer.data:
        print(f"\nQuaternion Normalization Check:")
        
        body_rot = buffer.data["body_rot"]
        root_rot = buffer.data["root_rot"]
        
        # Check a sample of quaternions
        sample_size = min(1000, body_rot.shape[0])
        sample_indices = np.random.choice(body_rot.shape[0], sample_size, replace=False)
        
        body_rot_norms = np.linalg.norm(body_rot[sample_indices].reshape(-1, 4), axis=1)
        root_rot_norms = np.linalg.norm(root_rot[sample_indices], axis=1)
        
        body_rot_norm_ok = np.allclose(body_rot_norms, 1.0, atol=0.01)
        root_rot_norm_ok = np.allclose(root_rot_norms, 1.0, atol=0.01)
        
        if body_rot_norm_ok:
            print(f"  ✓ body_rot: Quaternions properly normalized (mean norm: {body_rot_norms.mean():.4f})")
        else:
            print(f"  ✗ body_rot: Quaternions not properly normalized (mean norm: {body_rot_norms.mean():.4f})")
            has_invalid_data = True
        
        if root_rot_norm_ok:
            print(f"  ✓ root_rot: Quaternions properly normalized (mean norm: {root_rot_norms.mean():.4f})")
        else:
            print(f"  ✗ root_rot: Quaternions not properly normalized (mean norm: {root_rot_norms.mean():.4f})")
            has_invalid_data = True
    
    # Sample episode analysis
    print(f"\nSample Episode Analysis:")
    if buffer.n_episodes > 0:
        sample_ep = buffer.get_episode(0)
        print(f"  Episode 0 length: {len(sample_ep['act'])}")
        print(f"  Action range: [{sample_ep['act'].min():.3f}, {sample_ep['act'].max():.3f}]")
        print(f"  Joint pos range: [{sample_ep['joint_pos'].min():.3f}, {sample_ep['joint_pos'].max():.3f}]")
        print(f"  Root pos range: [{sample_ep['root_pos'].min():.3f}, {sample_ep['root_pos'].max():.3f}]")
    
    # Overall verdict
    print("\n" + "="*60)
    if all_fields_valid and not has_invalid_data:
        print("✓ Dataset verification PASSED")
        print("  Dataset is compatible with g1_offline_dataset.py")
        print("="*60)
        return True
    else:
        print("✗ Dataset verification FAILED")
        if not all_fields_valid:
            print("  - Some required fields are missing or have wrong shape")
        if has_invalid_data:
            print("  - Dataset contains invalid data (NaN, Inf, or unnormalized quaternions)")
        print("="*60)
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify G1 dataset structure")
    parser.add_argument("zarr_path", type=str, help="Path to zarr dataset file")
    args = parser.parse_args()
    
    success = verify_dataset(args.zarr_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
