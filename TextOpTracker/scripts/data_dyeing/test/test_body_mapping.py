"""
Test script to verify body index remapping between IsaacLab and MotionCLIP.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from body_index_mapping import (
    print_mapping_info,
    create_body_index_mapping,
    create_dof_index_mapping,
    remap_isaaclab_to_motionclip,
    remap_motionclip_to_isaaclab,
    ISAACLAB_BODY_NAMES,
    MOTIONCLIP_BODY_NAMES,
    ISAACLAB_DOF_NAMES,
    MOTIONCLIP_DOF_NAMES,
)
import numpy as np


def test_body_ordering():
    """Test that body name ordering is correct."""
    print("="*80)
    print("TEST 1: Body Name Ordering")
    print("="*80)
    
    print(f"\nIsaacLab ordering (alphabetical): {len(ISAACLAB_BODY_NAMES)} bodies")
    print(f"First 5: {ISAACLAB_BODY_NAMES[:5]}")
    print(f"Last 5: {ISAACLAB_BODY_NAMES[-5:]}")
    
    print(f"\nMotionCLIP ordering (semantic): {len(MOTIONCLIP_BODY_NAMES)} bodies")
    print(f"First 5: {MOTIONCLIP_BODY_NAMES[:5]}")
    print(f"Last 5: {MOTIONCLIP_BODY_NAMES[-5:]}")
    
    # Check if IsaacLab is truly alphabetical
    is_sorted = ISAACLAB_BODY_NAMES == sorted(MOTIONCLIP_BODY_NAMES)
    print(f"\nIsaacLab is alphabetically sorted: {'✓ PASS' if is_sorted else '✗ FAIL'}")
    
    # Check if all names match
    isaaclab_set = set(ISAACLAB_BODY_NAMES)
    motionclip_set = set(MOTIONCLIP_BODY_NAMES)
    all_match = isaaclab_set == motionclip_set
    print(f"All body names present in both: {'✓ PASS' if all_match else '✗ FAIL'}")
    
    if not all_match:
        print(f"  Missing in IsaacLab: {motionclip_set - isaaclab_set}")
        print(f"  Missing in MotionCLIP: {isaaclab_set - motionclip_set}")
    
    return is_sorted and all_match


def test_dof_ordering():
    """Test that DOF name ordering is correct."""
    print("\n" + "="*80)
    print("TEST 2: DOF Name Ordering")
    print("="*80)
    
    print(f"\nIsaacLab ordering (alphabetical): {len(ISAACLAB_DOF_NAMES)} DOFs")
    print(f"First 5: {ISAACLAB_DOF_NAMES[:5]}")
    print(f"Last 5: {ISAACLAB_DOF_NAMES[-5:]}")
    
    print(f"\nMotionCLIP ordering (semantic): {len(MOTIONCLIP_DOF_NAMES)} DOFs")
    print(f"First 5: {MOTIONCLIP_DOF_NAMES[:5]}")
    print(f"Last 5: {MOTIONCLIP_DOF_NAMES[-5:]}")
    
    # Check if IsaacLab is truly alphabetical
    is_sorted = ISAACLAB_DOF_NAMES == sorted(MOTIONCLIP_DOF_NAMES)
    print(f"\nIsaacLab is alphabetically sorted: {'✓ PASS' if is_sorted else '✗ FAIL'}")
    
    # Check if all names match
    isaaclab_set = set(ISAACLAB_DOF_NAMES)
    motionclip_set = set(MOTIONCLIP_DOF_NAMES)
    all_match = isaaclab_set == motionclip_set
    print(f"All DOF names present in both: {'✓ PASS' if all_match else '✗ FAIL'}")
    
    if not all_match:
        print(f"  Missing in IsaacLab: {motionclip_set - isaaclab_set}")
        print(f"  Missing in MotionCLIP: {isaaclab_set - motionclip_set}")
    
    return is_sorted and all_match


def test_remapping_roundtrip():
    """Test that remapping works correctly with round-trip."""
    print("\n" + "="*80)
    print("TEST 3: Remapping Round-Trip")
    print("="*80)
    
    # Create dummy data in IsaacLab format
    T = 100
    np.random.seed(42)
    
    isaaclab_data = {
        'body_pos': np.random.randn(T, 30, 3),
        'body_rot': np.random.randn(T, 30, 4),
        'body_lin_vel': np.random.randn(T, 30, 3),
        'body_ang_vel': np.random.randn(T, 30, 3),
        'joint_pos': np.random.randn(T, 29),
        'joint_vel': np.random.randn(T, 29),
    }
    
    print(f"\nOriginal IsaacLab data:")
    for key, val in isaaclab_data.items():
        print(f"  {key}: {val.shape}")
    
    # Remap to MotionCLIP
    motionclip_data = remap_isaaclab_to_motionclip(isaaclab_data)
    
    print(f"\nAfter remapping to MotionCLIP (same shapes, reordered):")
    for key, val in motionclip_data.items():
        print(f"  {key}: {val.shape}")
    
    # Check that data is different (indices were reordered)
    bodies_different = not np.allclose(isaaclab_data['body_pos'], motionclip_data['body_pos'])
    print(f"\nBody positions changed after remapping: {'✓ PASS' if bodies_different else '✗ FAIL'}")
    
    # Remap back to IsaacLab
    isaaclab_data_back = remap_motionclip_to_isaaclab(motionclip_data)
    
    print(f"\nAfter remapping back to IsaacLab:")
    for key, val in isaaclab_data_back.items():
        print(f"  {key}: {val.shape}")
    
    # Verify round-trip
    print(f"\nRound-trip verification:")
    all_pass = True
    for key in isaaclab_data.keys():
        matches = np.allclose(isaaclab_data[key], isaaclab_data_back[key])
        status = '✓ PASS' if matches else '✗ FAIL'
        print(f"  {key}: {status}")
        all_pass = all_pass and matches
    
    return all_pass and bodies_different


def test_specific_indices():
    """Test specific body index mappings."""
    print("\n" + "="*80)
    print("TEST 4: Specific Body Index Mapping")
    print("="*80)
    
    mapping = create_body_index_mapping()
    
    # Test pelvis (should be at index 0 in both)
    pelvis_mc_idx = MOTIONCLIP_BODY_NAMES.index('pelvis')
    pelvis_il_idx = ISAACLAB_BODY_NAMES.index('pelvis')
    pelvis_mapped_idx = mapping[pelvis_mc_idx]
    
    print(f"\nPelvis indices:")
    print(f"  MotionCLIP index: {pelvis_mc_idx}")
    print(f"  IsaacLab index: {pelvis_il_idx}")
    print(f"  Mapped index: {pelvis_mapped_idx}")
    print(f"  Mapping correct: {'✓ PASS' if pelvis_mapped_idx == pelvis_il_idx else '✗ FAIL'}")
    
    # Test a few more bodies
    test_bodies = [
        'left_knee_link',
        'right_shoulder_roll_link',
        'torso_link',
    ]
    
    all_correct = pelvis_mapped_idx == pelvis_il_idx
    
    for body_name in test_bodies:
        mc_idx = MOTIONCLIP_BODY_NAMES.index(body_name)
        il_idx = ISAACLAB_BODY_NAMES.index(body_name)
        mapped_idx = mapping[mc_idx]
        correct = mapped_idx == il_idx
        all_correct = all_correct and correct
        print(f"\n{body_name}:")
        print(f"  MotionCLIP idx: {mc_idx}, IsaacLab idx: {il_idx}, Mapped: {mapped_idx}")
        print(f"  {'✓ PASS' if correct else '✗ FAIL'}")
    
    return all_correct


def main():
    print("="*80)
    print("BODY INDEX REMAPPING TEST SUITE")
    print("="*80)
    
    # Run all tests
    results = []
    results.append(("Body Name Ordering", test_body_ordering()))
    results.append(("DOF Name Ordering", test_dof_ordering()))
    results.append(("Remapping Round-Trip", test_remapping_roundtrip()))
    results.append(("Specific Index Mapping", test_specific_indices()))
    
    # Print detailed mapping info
    print("\n" + "="*80)
    print("DETAILED MAPPING INFO")
    print("="*80)
    print_mapping_info()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:30s} {status}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Body index remapping is working correctly.")
    else:
        print("\n✗ Some tests failed. Please check the body/joint name lists.")
    
    print("="*80)
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
