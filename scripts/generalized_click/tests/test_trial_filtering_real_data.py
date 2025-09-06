"""Test trial filtering for MINT compatibility."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from gc_preprocessing import process_single_nwb
from trial_filtering import filter_trials_for_mint, extract_behavioral_features, cluster_behavioral_features
import numpy as np

def test_trial_filtering():
    """Test trial filtering for MINT compatibility on real data."""
    
    # Load data
    nwb_dir = Path("C:/Users/aqin/Downloads/GeneralizedClick/mint_pipeline_output/nwb_files")
    nwb_files = list(nwb_dir.glob("*.nwb"))[:1]
    
    if not nwb_files:
        print("No NWB files found for testing")
        return
    
    print(f"Testing trial filtering on {nwb_files[0].name}")
    print("=" * 60)
    
    # Extract trial data
    spikes, behavior, condition_ids, trial_data = process_single_nwb(nwb_files[0], verbose=False)
    
    # Check target distribution
    n_with_targets = sum(1 for trial in trial_data if trial['target'] is not None)
    n_without_targets = len(trial_data) - n_with_targets
    
    print(f"Original data:")
    print(f"  Total trials: {len(trial_data)}")
    print(f"  With targets: {n_with_targets}")
    print(f"  Without targets: {n_without_targets}")
    
    if n_without_targets == 0:
        print("\nNo trials without targets - artificially removing some targets for testing")
        # Artificially remove some targets for testing
        for i in range(0, min(50, len(trial_data)), 3):
            trial_data[i]['target'] = None
        
        n_with_targets = sum(1 for trial in trial_data if trial['target'] is not None)
        n_without_targets = len(trial_data) - n_with_targets
        print(f"After artificial removal:")
        print(f"  With targets: {n_with_targets}")
        print(f"  Without targets: {n_without_targets}")
    
    # Test feature extraction
    print(f"\n=== Testing Feature Extraction ===")
    
    # Debug: check a few trials without targets
    no_target_trials = [trial for trial in trial_data if trial['target'] is None]
    print(f"Checking first few target-less trials:")
    
    # Group by task and epoch type
    no_target_types = {}
    for trial in no_target_trials[:20]:  # Check first 20
        key = (trial['task'], trial['epoch_type'])
        if key not in no_target_types:
            no_target_types[key] = []
        no_target_types[key].append(trial)
    
    for (task, epoch_type), trials in no_target_types.items():
        pos_shape = trials[0]['position'].shape if trials[0]['position'] is not None else "None"
        vel_shape = trials[0]['velocity'].shape if trials[0]['velocity'] is not None else "None"
        print(f"  {task}/{epoch_type}: {len(trials)} trials, position={pos_shape}, velocity={vel_shape}")
        
        # Check if any have actual movement data
        if len(trials) > 0 and trials[0]['position'] is not None and trials[0]['position'].shape[0] > 0:
            pos_range = np.ptp(trials[0]['position'], axis=0)
            vel_range = np.ptp(trials[0]['velocity'], axis=0)
            print(f"    Movement range: pos={pos_range}, vel={vel_range}")
    
    feature_data = extract_behavioral_features(trial_data, verbose=True)
    
    if len(feature_data['trial_indices']) == 0:
        print("No features extracted - cannot test clustering")
        return
    
    # Test clustering
    print(f"\n=== Testing Behavioral Clustering ===")
    cluster_labels, clustering_info = cluster_behavioral_features(
        feature_data['features'], 
        verbose=True
    )
    
    # Test full pipeline
    print(f"\n=== Testing Full Pipeline (with filtering) ===")
    print(f"Original trial count: {len(trial_data)}")
    
    filtered_trial_data = filter_trials_for_mint(trial_data, verbose=True)
    
    # Analyze results
    print(f"\n=== Results Analysis ===")
    print(f"Trials after filtering: {len(filtered_trial_data)} (kept {len(filtered_trial_data)/len(trial_data)*100:.1f}%)")
    
    n_real_targets = sum(1 for trial in filtered_trial_data 
                        if trial['target'] is not None and not trial.get('is_pseudo_target', False))
    n_pseudo_targets = sum(1 for trial in filtered_trial_data 
                          if trial.get('is_pseudo_target', False))
    n_no_targets = sum(1 for trial in filtered_trial_data 
                      if trial['target'] is None)
    
    print(f"Final results:")
    print(f"  Real targets: {n_real_targets}")
    print(f"  Pseudo-targets: {n_pseudo_targets}")
    print(f"  No targets: {n_no_targets}")
    
    # Show what was discarded
    print(f"\nDiscarded trials analysis:")
    discarded_count = len(trial_data) - len(filtered_trial_data)
    print(f"  Total discarded: {discarded_count}")
    
    # Show some examples
    print(f"\nKept trial examples:")
    for i, trial in enumerate(filtered_trial_data[:5]):
        target_type = "real" if not trial.get('is_pseudo_target', False) else "pseudo"
        print(f"  Trial {i}: task={trial['task']}, epoch={trial['epoch_type']}, target_type={target_type}")
    
    # Test that we can now create condition keys
    print(f"\n=== Testing Condition Key Generation ===")
    condition_keys = []
    for trial in filtered_trial_data:
        task = trial['task']
        epoch_type = trial['epoch_type']
        target = trial['target']
        
        if target is not None:
            if isinstance(target, (list, tuple)) and len(target) >= 2:
                rounded_target = tuple(np.round(target[:2], decimals=2))
            else:
                rounded_target = target
        else:
            rounded_target = "NO_TARGET"
        
        condition_key = (task, epoch_type, rounded_target)
        condition_keys.append(condition_key)
    
    unique_conditions = list(set(condition_keys))
    
    print(f"Generated {len(unique_conditions)} unique conditions from filtered trials")
    print(f"First 10 conditions:")
    for i, cond in enumerate(unique_conditions[:10]):
        count = condition_keys.count(cond)
        print(f"  {i}: {cond} ({count} trials)")
    
    # Test sorting
    try:
        def sort_key(condition_tuple):
            task, epoch_type, target = condition_tuple
            if target == "NO_TARGET":
                target_sort = (0,)
            elif isinstance(target, tuple):
                target_sort = (1,) + target
            else:
                target_sort = (2, str(target))
            return (task, epoch_type, target_sort)
        
        unique_conditions.sort(key=sort_key)
        print(f"\nSorting successful!")
        print(f"First 5 sorted conditions:")
        for i, cond in enumerate(unique_conditions[:5]):
            print(f"  {i}: {cond}")
        
        return True
        
    except Exception as e:
        print(f"\nSorting failed: {e}")
        return False


if __name__ == "__main__":
    success = test_trial_filtering()
    if success:
        print(f"\nSUCCESS: Trial filtering pipeline working!")
    else:
        print(f"\nFAIL: Issues with trial filtering pipeline")