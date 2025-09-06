"""Condition balancing for MINT training.

Balances condition representation by removing sparse conditions,
capping excessive trials, and limiting total conditions.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from collections import Counter


def balance_conditions(spikes: np.ndarray, 
                      behavior: np.ndarray, 
                      condition_ids: np.ndarray,
                      trial_data: List[Dict],
                      max_trials_per_condition: Optional[int] = None,  # No cap by default (following original MINT paper)
                      min_trials_per_condition: int = 5,  # MINT paper requirement
                      max_total_conditions: int = 100,
                      verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict], Dict]:
    """Balance condition representation for optimal MINT training.
    
    Args:
        spikes: Neural spike data [trials, neurons, time]
        behavior: Behavioral data [trials, behavior_dims, time]
        condition_ids: Condition IDs [trials]
        trial_data: List of trial dictionaries
        max_trials_per_condition: Maximum trials to keep per condition (None = no cap)
        min_trials_per_condition: Minimum trials required per condition
        max_total_conditions: Maximum total conditions to keep
        verbose: Whether to print progress
        
    Returns:
        Tuple of (balanced_spikes, balanced_behavior, balanced_condition_ids, balanced_trial_data, balancing_stats)
    """
    # Count trials per condition
    condition_counts = Counter(condition_ids)
    unique_conditions = list(condition_counts.keys())
    
    if verbose:
        print(f"\n    Condition balancing analysis:")
        print(f"      Total conditions: {len(unique_conditions)}")
        print(f"      Total trials: {len(condition_ids)}")
        print(f"      Trials per condition: min={min(condition_counts.values())}, max={max(condition_counts.values())}, mean={np.mean(list(condition_counts.values())):.1f}")
    
    # Filter out conditions with too few trials
    valid_conditions = [cond for cond, count in condition_counts.items() if count >= min_trials_per_condition]
    removed_conditions = [cond for cond in unique_conditions if cond not in valid_conditions]
    
    if verbose and removed_conditions:
        print(f"      Removing {len(removed_conditions)} conditions with < {min_trials_per_condition} trials")
        for cond in removed_conditions[:10]:  # Show first 10
            print(f"        Condition {cond}: {condition_counts[cond]} trials")
        if len(removed_conditions) > 10:
            print(f"        ... and {len(removed_conditions) - 10} more")
    
    # If still too many conditions, keep the ones with the most trials
    if len(valid_conditions) > max_total_conditions:
        # Sort by trial count (descending)
        valid_conditions.sort(key=lambda x: condition_counts[x], reverse=True)
        kept_conditions = valid_conditions[:max_total_conditions]
        dropped_conditions = valid_conditions[max_total_conditions:]
        
        if verbose:
            print(f"      Too many conditions ({len(valid_conditions)}), keeping top {max_total_conditions} by trial count")
            print(f"      Dropped {len(dropped_conditions)} conditions with fewer trials")
        
        valid_conditions = kept_conditions
    
    # Balance trials within each valid condition
    balanced_indices = []
    condition_balancing_stats = {}
    
    for condition in valid_conditions:
        # Get all trial indices for this condition
        condition_mask = condition_ids == condition
        condition_indices = np.where(condition_mask)[0]
        
        # Limit to max_trials_per_condition if specified
        if max_trials_per_condition is not None and len(condition_indices) > max_trials_per_condition:
            # Randomly sample without replacement
            np.random.seed(42)  # For reproducibility
            selected_indices = np.random.choice(
                condition_indices, 
                size=max_trials_per_condition, 
                replace=False
            )
        else:
            selected_indices = condition_indices
        
        balanced_indices.extend(selected_indices)
        
        condition_balancing_stats[condition] = {
            'original_count': len(condition_indices),
            'final_count': len(selected_indices),
            'subsampled': max_trials_per_condition is not None and len(condition_indices) > max_trials_per_condition
        }
    
    # Convert to array and sort for consistent ordering
    balanced_indices = np.array(balanced_indices)
    balanced_indices.sort()
    
    # Extract balanced data
    balanced_spikes = spikes[balanced_indices]
    balanced_behavior = behavior[balanced_indices]
    balanced_condition_ids = condition_ids[balanced_indices]
    
    # Handle case where trial_data might be shorter due to filtering
    if len(trial_data) == len(spikes):
        balanced_trial_data = [trial_data[i] for i in balanced_indices]
    else:
        # trial_data was filtered, so indices don't align - keep original trial_data
        balanced_trial_data = trial_data
    
    # Update condition IDs to be consecutive (0, 1, 2, ...)
    old_to_new_condition_map = {old_cond: new_cond for new_cond, old_cond in enumerate(sorted(valid_conditions))}
    remapped_condition_ids = np.array([old_to_new_condition_map[cond] for cond in balanced_condition_ids])
    
    # Create balancing statistics
    balancing_stats = {
        'original_trials': len(condition_ids),
        'final_trials': len(balanced_indices),
        'original_conditions': len(unique_conditions),
        'final_conditions': len(valid_conditions),
        'removed_conditions': removed_conditions,
        'condition_stats': condition_balancing_stats,
        'condition_mapping': old_to_new_condition_map,
        'trials_removed': len(condition_ids) - len(balanced_indices),
        'subsampled_conditions': sum(1 for stats in condition_balancing_stats.values() if stats['subsampled'])
    }
    
    if verbose:
        print(f"\n    Balancing results:")
        print(f"      Final conditions: {len(valid_conditions)} (removed {len(unique_conditions) - len(valid_conditions)})")
        print(f"      Final trials: {len(balanced_indices)} (removed {len(condition_ids) - len(balanced_indices)})")
        print(f"      Subsampled conditions: {balancing_stats['subsampled_conditions']}")
        
        # Show final distribution
        final_counts = Counter(remapped_condition_ids)
        print(f"      Final trials per condition: min={min(final_counts.values())}, max={max(final_counts.values())}, mean={np.mean(list(final_counts.values())):.1f}")
    
    return balanced_spikes, balanced_behavior, remapped_condition_ids, balanced_trial_data, balancing_stats


def validate_condition_balance(condition_ids: np.ndarray, 
                              target_min: int = 5,  # MINT paper requirement
                              target_max: int = 30000,
                              verbose: bool = True) -> Dict:
    """Validate that conditions are reasonably balanced.
    
    Args:
        condition_ids: Array of condition IDs
        target_min: Target minimum trials per condition
        target_max: Target maximum trials per condition
        verbose: Whether to print results
        
    Returns:
        Dictionary with validation results
    """
    condition_counts = Counter(condition_ids)
    counts = list(condition_counts.values())
    
    validation_results = {
        'total_conditions': len(condition_counts),
        'total_trials': len(condition_ids),
        'min_trials': min(counts),
        'max_trials': max(counts),
        'mean_trials': np.mean(counts),
        'std_trials': np.std(counts),
        'conditions_below_min': sum(1 for c in counts if c < target_min),
        'conditions_above_max': sum(1 for c in counts if c > target_max),
        'well_balanced': True
    }
    
    # Check if balancing is needed
    if validation_results['conditions_below_min'] > 0 or validation_results['conditions_above_max'] > 0:
        validation_results['well_balanced'] = False
    
    if validation_results['std_trials'] > validation_results['mean_trials'] * 0.5:  # High variance
        validation_results['well_balanced'] = False
    
    if verbose:
        print(f"\n    Condition balance validation:")
        print(f"      Total conditions: {validation_results['total_conditions']}")
        print(f"      Trials per condition: min={validation_results['min_trials']}, max={validation_results['max_trials']}, mean={validation_results['mean_trials']:.1f}, std={validation_results['std_trials']:.1f}")
        print(f"      Conditions below target min ({target_min}): {validation_results['conditions_below_min']}")
        print(f"      Conditions above target max ({target_max}): {validation_results['conditions_above_max']}")
        print(f"      Well balanced: {validation_results['well_balanced']}")
    
    return validation_results