"""Quick data quality check without running MINT decoder."""

import numpy as np
from pathlib import Path
from gc_preprocessing import process_single_nwb


def quick_data_check():
    """Quick check of a single NWB file's data quality."""
    
    # Load just one file for speed
    nwb_dir = Path("C:/Users/aqin/Downloads/GeneralizedClick/mint_pipeline_output/nwb_files")
    nwb_files = list(nwb_dir.glob("*.nwb"))[:1]
    
    if not nwb_files:
        print("No NWB files found")
        return
    
    print(f"Checking data quality from {nwb_files[0].name}")
    print("=" * 50)
    
    # Extract data
    spikes, behavior, condition_ids, trial_data = process_single_nwb(nwb_files[0], verbose=True)
    
    print(f"\n=== EXTRACTED DATA ANALYSIS ===")
    print(f"Shapes: spikes={spikes.shape}, behavior={behavior.shape}, conditions={condition_ids.shape}")
    
    # Check spike data
    print(f"\nSpike data analysis:")
    print(f"  Data type: {spikes.dtype}")
    print(f"  Range: [{spikes.min():.6f}, {spikes.max():.6f}]")
    print(f"  Mean: {spikes.mean():.6f}")
    print(f"  Std: {spikes.std():.6f}")
    print(f"  NaNs: {np.isnan(spikes).sum()}")
    print(f"  Infs: {np.isinf(spikes).sum()}")
    print(f"  Zeros: {(spikes == 0).sum()} / {spikes.size} ({(spikes == 0).sum()/spikes.size*100:.1f}%)")
    
    # Check if data looks like spike counts vs rates
    if spikes.max() > 100:
        print(f"  ISSUE: Very high values (max={spikes.max():.1f}) - might be spike counts not rates")
    if spikes.mean() < 0.01:
        print(f"  ISSUE: Very low mean rate ({spikes.mean():.6f})")
    
    # Check behavior data  
    print(f"\nBehavior data analysis:")
    print(f"  Data type: {behavior.dtype}")
    print(f"  Range: [{behavior.min():.6f}, {behavior.max():.6f}]")
    print(f"  Mean: {behavior.mean():.6f}")
    print(f"  NaNs: {np.isnan(behavior).sum()}")
    print(f"  Zeros: {(behavior == 0).sum()} / {behavior.size} ({(behavior == 0).sum()/behavior.size*100:.1f}%)")
    
    # Check each behavior dimension
    for i, name in enumerate(['pos_x', 'pos_y', 'vel_x', 'vel_y']):
        data = behavior[:, i, :]
        print(f"  {name}: range=[{data.min():.3f}, {data.max():.3f}], mean={data.mean():.3f}, std={data.std():.3f}")
    
    # Check conditions
    unique_conds, counts = np.unique(condition_ids, return_counts=True)
    print(f"\nCondition analysis:")
    print(f"  Unique conditions: {len(unique_conds)}")
    print(f"  Trials per condition: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")
    print(f"  Conditions with <5 trials: {(counts < 5).sum()}")
    
    # Check for specific MINT issues
    print(f"\n=== MINT COMPATIBILITY CHECKS ===")
    
    # 1. Check if spike data is in the right format
    spike_issues = []
    if spikes.min() < 0:
        spike_issues.append("Negative spike values (should be rates >= 0)")
    if spikes.max() > 1000:
        spike_issues.append(f"Very high spike values (max={spikes.max():.1f} - should be rates in Hz)")
    if spikes.mean() < 0.001:
        spike_issues.append(f"Extremely low spike rates (mean={spikes.mean():.6f})")
    if np.var(spikes) < 1e-10:
        spike_issues.append("Nearly constant spike data")
    
    # 2. Check neural data quality
    zero_neurons = np.sum(np.all(spikes == 0, axis=(0, 2)))
    constant_neurons = np.sum(np.var(spikes, axis=(0, 2)) < 1e-10)
    
    if zero_neurons > 0:
        spike_issues.append(f"{zero_neurons} neurons with all-zero activity")
    if constant_neurons > 0:
        spike_issues.append(f"{constant_neurons} neurons with constant activity")
    
    # 3. Check behavior data quality
    behavior_issues = []
    zero_behavior_trials = np.sum(np.all(behavior == 0, axis=(1, 2)))
    if zero_behavior_trials > 0:
        behavior_issues.append(f"{zero_behavior_trials} trials with all-zero behavior")
    
    for i, name in enumerate(['pos_x', 'pos_y', 'vel_x', 'vel_y']):
        data = behavior[:, i, :]
        if np.var(data) < 1e-10:
            behavior_issues.append(f"{name} has constant values")
    
    # 4. Check temporal structure
    time_issues = []
    if spikes.shape[2] != behavior.shape[2]:
        time_issues.append("Spike and behavior time dimensions don't match")
    if spikes.shape[2] < 10:
        time_issues.append(f"Very short trials ({spikes.shape[2]} time bins)")
    
    # Report issues
    all_issues = spike_issues + behavior_issues + time_issues
    
    if all_issues:
        print(f"ISSUES FOUND:")
        for issue in all_issues:
            print(f"  - {issue}")
    else:
        print(f"No obvious data quality issues found")
    
    # Check a few example trials in detail
    print(f"\n=== SAMPLE TRIAL ANALYSIS ===")
    for trial_idx in [0, spikes.shape[0]//2, -1]:
        if trial_idx < 0:
            trial_idx = spikes.shape[0] + trial_idx
            
        trial_spikes = spikes[trial_idx]  # (neurons, time)
        trial_behavior = behavior[trial_idx]  # (4, time)
        
        print(f"Trial {trial_idx}:")
        print(f"  Spike activity: {trial_spikes.sum():.1f} total, {trial_spikes.mean():.3f} mean rate")
        print(f"  Active neurons: {np.sum(trial_spikes.sum(axis=1) > 0)} / {trial_spikes.shape[0]}")
        print(f"  Behavior range: pos=[{trial_behavior[:2].min():.3f}, {trial_behavior[:2].max():.3f}], vel=[{trial_behavior[2:].min():.3f}, {trial_behavior[2:].max():.3f}]")
        
        # Check if behavior changes over time
        pos_movement = np.sqrt(np.sum(np.diff(trial_behavior[:2], axis=1)**2))
        print(f"  Total movement: {pos_movement:.3f}")
    
    return len(all_issues) == 0


if __name__ == "__main__":
    success = quick_data_check()
    print(f"\nData check: {'PASSED' if success else 'FAILED'}")