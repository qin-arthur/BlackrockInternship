"""Test the new trial filtering approach."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from trial_filtering import should_discard_trial, filter_trials_for_mint


def test_trial_filtering():
    """Test that trials are properly filtered based on MINT compatibility."""
    
    print("=== Testing Trial Filtering Logic ===")
    
    # Create test trials with different characteristics
    test_trials = [
        # Good trial with target and behavior
        {
            'task': 'Click', 
            'epoch_type': 'Click', 
            'target': (0.1, 0.2),
            'position': np.random.randn(50, 2),
            'velocity': np.random.randn(50, 2)
        },
        
        # Good trial without target but with behavior (should get pseudo-target)
        {
            'task': 'Click', 
            'epoch_type': 'Click', 
            'target': None,
            'position': np.random.randn(50, 2),
            'velocity': np.random.randn(50, 2)
        },
        
        # Bad trial - InterTrial epoch (should be discarded)
        {
            'task': 'Click', 
            'epoch_type': 'InterTrial', 
            'target': None,
            'position': np.random.randn(50, 2),
            'velocity': np.random.randn(50, 2)
        },
        
        # Bad trial - FailSafe epoch (should be discarded)
        {
            'task': 'Click', 
            'epoch_type': 'FailSafe', 
            'target': None,
            'position': np.random.randn(10, 2),
            'velocity': np.random.randn(10, 2)
        },
        
        # Bad trial - Calibration task (should be discarded)
        {
            'task': 'OrthoCalibration', 
            'epoch_type': 'Calibration', 
            'target': (0.0, 0.0),
            'position': np.random.randn(20, 2),
            'velocity': np.random.randn(20, 2)
        },
        
        # Bad trial - no behavioral data (should be discarded)
        {
            'task': 'Click', 
            'epoch_type': 'Click', 
            'target': None,
            'position': np.zeros((3, 2)),  # Too short and no movement
            'velocity': np.zeros((3, 2))
        },
        
        # Good trial - Reach with behavior
        {
            'task': 'Reach', 
            'epoch_type': 'Reach', 
            'target': None,
            'position': np.random.randn(40, 2),
            'velocity': np.random.randn(40, 2)
        }
    ]
    
    print(f"Testing {len(test_trials)} trials...")
    
    # Test individual filtering decisions
    print(f"\nIndividual filtering decisions:")
    for i, trial in enumerate(test_trials):
        should_discard, reason = should_discard_trial(trial)
        status = "DISCARD" if should_discard else "KEEP"
        print(f"  Trial {i} ({trial['task']}/{trial['epoch_type']}): {status} - {reason}")
    
    # Test full filtering pipeline
    print(f"\n=== Testing Full Filtering Pipeline ===")
    
    filtered_trials = filter_trials_for_mint(test_trials, verbose=True)
    
    print(f"\nFiltering results:")
    print(f"  Original: {len(test_trials)} trials")
    print(f"  Filtered: {len(filtered_trials)} trials")
    print(f"  Kept: {len(filtered_trials)/len(test_trials)*100:.1f}%")
    
    # Verify expected results
    expected_kept = 3  # Trials 0, 1, 6 should be kept
    expected_discarded = ['InterTrial', 'FailSafe', 'Calibration']
    
    if len(filtered_trials) == expected_kept:
        print(f"PASS: Correct number of trials kept ({expected_kept})")
    else:
        print(f"FAIL: Expected {expected_kept} trials, got {len(filtered_trials)}")
        return False
    
    # Check that kept trials are the right ones
    kept_epoch_types = [trial['epoch_type'] for trial in filtered_trials]
    expected_kept_epochs = ['Click', 'Click', 'Reach']
    
    if sorted(kept_epoch_types) == sorted(expected_kept_epochs):
        print(f"PASS: Correct epoch types kept: {kept_epoch_types}")
    else:
        print(f"FAIL: Expected epochs {expected_kept_epochs}, got {kept_epoch_types}")
        return False
    
    # Check pseudo-target assignment
    pseudo_targets = [trial for trial in filtered_trials if trial.get('is_pseudo_target', False)]
    expected_pseudo = 2  # Trials 1 and 6 should get pseudo-targets
    
    if len(pseudo_targets) == expected_pseudo:
        print(f"PASS: Correct number of pseudo-targets assigned ({expected_pseudo})")
    else:
        print(f"FAIL: Expected {expected_pseudo} pseudo-targets, got {len(pseudo_targets)}")
        return False
    
    print(f"\nSUCCESS: Trial filtering working correctly!")
    print(f"  - Properly discards InterTrial, FailSafe, Calibration epochs")
    print(f"  - Keeps trials with behavioral data and movement epochs")
    print(f"  - Assigns pseudo-targets to targetless movement trials")
    print(f"  - Compatible with MINT's assumptions about trajectories")
    
    return True


if __name__ == "__main__":
    success = test_trial_filtering()
    if not success:
        print(f"\nFAILURE: Trial filtering has issues")
        exit(1)