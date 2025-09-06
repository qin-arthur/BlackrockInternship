"""Test adaptive precision for target rounding."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from gc_preprocessing import process_single_nwb, compute_adaptive_target_precision, create_unified_condition_mapping
import numpy as np

# Test on one file
nwb_dir = Path("C:/Users/aqin/Downloads/GeneralizedClick/mint_pipeline_output/nwb_files")
nwb_files = list(nwb_dir.glob("*.nwb"))[:1]

if nwb_files:
    print(f"Testing adaptive precision on {nwb_files[0].name}")
    
    # Extract data
    spikes, behavior, condition_ids, trial_data = process_single_nwb(nwb_files[0], verbose=False)
    
    print(f"\nExtracted {len(trial_data)} trials")
    
    # Test adaptive precision computation
    print(f"\n=== Testing Adaptive Precision ===")
    adaptive_precision = compute_adaptive_target_precision(trial_data, verbose=True)
    
    # Test unified condition mapping with adaptive precision
    print(f"\n=== Testing Unified Condition Mapping ===")
    unified_mapping = create_unified_condition_mapping(trial_data, verbose=True)
    
    # Compare with fixed precision
    print(f"\n=== Comparison with Fixed Precision (0.1) ===")
    
    # Count conditions with fixed precision
    fixed_condition_keys = []
    for trial in trial_data:
        task = trial['task']
        epoch_type = trial['epoch_type']
        target = trial['target']
        
        if target is not None:
            rounded_target = tuple(np.round(target, decimals=1))
        else:
            rounded_target = None
        
        condition_key = (task, epoch_type, rounded_target)
        fixed_condition_keys.append(condition_key)
    
    fixed_unique_keys = list(set(fixed_condition_keys))
    
    print(f"Fixed precision (0.1): {len(fixed_unique_keys)} unique conditions")
    print(f"Adaptive precision ({adaptive_precision}): {len(unified_mapping['unique_keys'])} unique conditions")
    
    # Show some example target clusterings
    print(f"\n=== Example Target Clustering ===")
    targets_with_tasks = []
    for trial in trial_data:
        if trial['target'] is not None:
            targets_with_tasks.append((trial['target'], trial['task'], trial['epoch_type']))
    
    if targets_with_tasks:
        print(f"First 10 targets and their rounded versions:")
        for i, (target, task, epoch_type) in enumerate(targets_with_tasks[:10]):
            target_array = np.array(target)
            if len(target_array) >= 2:
                original = f"({target_array[0]:.3f}, {target_array[1]:.3f})"
                rounded_x = float(np.round(target_array[0] / adaptive_precision) * adaptive_precision)
                rounded_y = float(np.round(target_array[1] / adaptive_precision) * adaptive_precision)
                rounded = f"({rounded_x:.3f}, {rounded_y:.3f})"
                print(f"  {task}/{epoch_type}: {original} -> {rounded}")
    
else:
    print("No NWB files found")