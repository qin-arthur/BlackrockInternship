"""Clean neural and behavioral data for MINT compatibility."""

import numpy as np
from typing import Tuple, List, Dict


def remove_dead_neurons(spikes: np.ndarray, threshold: float = 1e-6, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Remove neurons with no or minimal activity.
    
    Args:
        spikes: Neural data (trials, neurons, time)
        threshold: Minimum variance threshold for active neurons
        verbose: Whether to print results
        
    Returns:
        Tuple of (cleaned_spikes, active_neuron_indices)
    """
    # Calculate variance across trials and time for each neuron
    neuron_variance = np.var(spikes, axis=(0, 2))
    
    # Find active neurons
    active_mask = neuron_variance > threshold
    active_indices = np.where(active_mask)[0]
    
    # Remove dead neurons
    cleaned_spikes = spikes[:, active_mask, :]
    
    if verbose:
        n_removed = np.sum(~active_mask)
        print(f"    Removed {n_removed} dead neurons (kept {len(active_indices)} active neurons)")
        print(f"    Active neuron variance range: [{neuron_variance[active_mask].min():.6f}, {neuron_variance[active_mask].max():.6f}]")
    
    return cleaned_spikes, active_indices


def remove_static_trials(spikes: np.ndarray, behavior: np.ndarray, condition_ids: np.ndarray, 
                        trial_data: List[Dict], min_movement: float = 0.01, 
                        verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    """Remove trials with no meaningful behavior or neural activity.
    
    Args:
        spikes: Neural data (trials, neurons, time) 
        behavior: Behavioral data (trials, 4, time)
        condition_ids: Condition IDs (trials,)
        trial_data: Trial metadata
        min_movement: Minimum total movement to keep trial
        verbose: Whether to print results
        
    Returns:
        Tuple of cleaned (spikes, behavior, condition_ids, trial_data)
    """
    keep_mask = []
    
    for i in range(spikes.shape[0]):
        trial_spikes = spikes[i]
        trial_behavior = behavior[i]
        
        # Check for minimal neural activity
        has_neural_activity = np.sum(trial_spikes) > 0
        
        # Check for behavioral movement (position change)
        position = trial_behavior[:2, :]  # x, y position
        total_movement = np.sum(np.sqrt(np.sum(np.diff(position, axis=1)**2, axis=0)))
        has_movement = total_movement > min_movement
        
        # Check for non-constant behavior
        behavior_variance = np.var(trial_behavior)
        has_behavior_variance = behavior_variance > 1e-8
        
        # Keep trial if it has activity and either movement or behavior variance
        keep = has_neural_activity and (has_movement or has_behavior_variance)
        keep_mask.append(keep)
    
    keep_mask = np.array(keep_mask)
    
    # Filter data
    cleaned_spikes = spikes[keep_mask]
    cleaned_behavior = behavior[keep_mask] 
    cleaned_condition_ids = condition_ids[keep_mask]
    cleaned_trial_data = [trial_data[i] for i in np.where(keep_mask)[0]]
    
    if verbose:
        n_removed = np.sum(~keep_mask)
        print(f"    Removed {n_removed} static trials (kept {np.sum(keep_mask)} active trials)")
    
    return cleaned_spikes, cleaned_behavior, cleaned_condition_ids, cleaned_trial_data


def smooth_sparse_data(spikes: np.ndarray, sigma: float = 1.0, verbose: bool = True) -> np.ndarray:
    """Apply light smoothing to very sparse neural data.
    
    Args:
        spikes: Neural data (trials, neurons, time)
        sigma: Gaussian smoothing sigma in time bins
        verbose: Whether to print results
        
    Returns:
        Smoothed neural data
    """
    from scipy.ndimage import gaussian_filter1d
    
    # Apply Gaussian smoothing along time axis
    smoothed_spikes = np.zeros_like(spikes)
    
    for trial in range(spikes.shape[0]):
        for neuron in range(spikes.shape[1]):
            smoothed_spikes[trial, neuron, :] = gaussian_filter1d(
                spikes[trial, neuron, :], 
                sigma=sigma, 
                mode='constant'
            )
    
    if verbose:
        original_sparsity = (spikes == 0).sum() / spikes.size * 100
        new_sparsity = (smoothed_spikes < 1e-6).sum() / smoothed_spikes.size * 100
        print(f"    Smoothing reduced sparsity from {original_sparsity:.1f}% to {new_sparsity:.1f}%")
    
    return smoothed_spikes


def balance_neural_activity(spikes: np.ndarray, max_rate_ratio: float = 100.0, 
                           verbose: bool = True) -> np.ndarray:
    """Balance neural activity to prevent extreme rate differences.
    
    Args:
        spikes: Neural data (trials, neurons, time)
        max_rate_ratio: Maximum ratio between highest and lowest active neuron rates
        verbose: Whether to print results
        
    Returns:
        Balanced neural data
    """
    # Calculate mean rate per neuron
    neuron_rates = np.mean(spikes, axis=(0, 2))
    
    # Find active neurons (rate > 0)
    active_mask = neuron_rates > 0
    if np.sum(active_mask) == 0:
        return spikes
    
    active_rates = neuron_rates[active_mask]
    min_rate = np.min(active_rates)
    max_rate = np.max(active_rates)
    
    if max_rate / min_rate > max_rate_ratio:
        # Cap maximum rates
        rate_cap = min_rate * max_rate_ratio
        
        balanced_spikes = spikes.astype(np.float64)
        for neuron in range(spikes.shape[1]):
            if neuron_rates[neuron] > rate_cap:
                scale_factor = rate_cap / neuron_rates[neuron]
                balanced_spikes[:, neuron, :] *= scale_factor
        
        if verbose:
            new_max_rate = np.max(np.mean(balanced_spikes, axis=(0, 2)))
            print(f"    Capped neural rates: {max_rate:.3f} -> {new_max_rate:.3f} Hz")
        
        return balanced_spikes
    
    return spikes


def clean_data_for_mint(spikes: np.ndarray, behavior: np.ndarray, condition_ids: np.ndarray,
                       trial_data: List[Dict], skip_neuron_removal: bool = True, 
                       verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    """Data cleaning for MINT compatibility - FIXED version without harmful regularization.
    
    Args:
        spikes: Neural data (trials, neurons, time)
        behavior: Behavioral data (trials, 4, time) 
        condition_ids: Condition IDs (trials,)
        trial_data: Trial metadata
        skip_neuron_removal: Always True in new architecture (neuron removal done globally)
        verbose: Whether to print cleaning steps
        
    Returns:
        Tuple of cleaned (spikes, behavior, condition_ids, trial_data)
    """
    if verbose:
        print(f"  Data cleaning for MINT compatibility:")
        print(f"    Original: {spikes.shape[0]} trials, {spikes.shape[1]} neurons")
    
    # 1. Neuron removal is done globally across all sessions
    cleaned_spikes = spikes
    
    # 2. Remove static trials
    cleaned_spikes, cleaned_behavior, cleaned_condition_ids, cleaned_trial_data = remove_static_trials(
        cleaned_spikes, behavior, condition_ids, trial_data, verbose=verbose
    )
    
    # 3. Light smoothing for very sparse data
    sparsity = (cleaned_spikes == 0).sum() / cleaned_spikes.size * 100
    if sparsity > 90:  # Very sparse data
        cleaned_spikes = smooth_sparse_data(cleaned_spikes, sigma=1.5, verbose=verbose)
    
    # 4. NO REGULARIZATION - This was causing NaN issues!
    # The original code added 1e-8 to spikes and 1e-10 to behavior
    # This is harmful because:
    # - It changes the scale of the data
    # - MINT expects actual spike counts/rates
    # - The tiny values can cause numerical instabilities
    
    if verbose:
        final_sparsity = (cleaned_spikes < 1e-6).sum() / cleaned_spikes.size * 100
        print(f"    Final: {cleaned_spikes.shape[0]} trials, {cleaned_spikes.shape[1]} neurons")
        print(f"    Final sparsity: {final_sparsity:.1f}%")
        print(f"    Neural rate range: [{cleaned_spikes.min():.6f}, {cleaned_spikes.max():.6f}]")
    
    return cleaned_spikes, cleaned_behavior, cleaned_condition_ids, cleaned_trial_data