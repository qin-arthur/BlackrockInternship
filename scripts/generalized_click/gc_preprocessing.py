"""NWB to MINT preprocessing pipeline.

Processes NWB files for MINT decoder training with trial filtering,
unified condition mapping, and movement-based alignment.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Union, Tuple
from pynwb import NWBHDF5IO
from movement_utils import compute_movement_windows
from trial_filtering import filter_trials_for_mint
import pandas as pd


def process_nwb_directory(nwb_dir: Union[str, Path], 
                         output_dir: Optional[Union[str, Path]] = None,
                         merge_sessions: bool = True,
                         verbose: bool = True) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Process all NWB files in a directory to extract MINT-compatible data.
    
    Args:
        nwb_dir: Directory containing NWB files
        output_dir: Optional directory to save processed data
        merge_sessions: If True, merge all sessions into single arrays with unified conditions
        verbose: Whether to print progress
        
    Returns:
        Dictionary mapping session names to (spikes, behavior, condition_ids) tuples
    """
    nwb_dir = Path(nwb_dir)
    nwb_files = list(nwb_dir.glob("*.nwb"))
    
    if not nwb_files:
        raise ValueError(f"No NWB files found in {nwb_dir}")
    
    if verbose:
        print(f"Found {len(nwb_files)} NWB files in {nwb_dir}")
    
    all_sessions = {}
    all_trial_data = []  # Collect all trial data for unified condition mapping
    
    # First pass: extract all trial data to build unified condition mapping
    if verbose:
        print(f"\nFirst pass: extracting trial data for unified condition mapping...")
    
    for nwb_file in nwb_files:
        if verbose:
            print(f"\nProcessing {nwb_file.name}...")
        
        try:
            spikes, behavior, condition_ids, trial_data = process_single_nwb(nwb_file, verbose=verbose)
            session_name = nwb_file.stem
            
            # Store raw data without final condition IDs yet
            all_sessions[session_name] = (spikes, behavior, trial_data)  # Store trial_data instead of condition_ids
            all_trial_data.extend(trial_data)  # Collect all trial data
            
                
        except Exception as e:
            if verbose:
                print(f"  ERROR processing {nwb_file.name}: {e}")
            continue
    
    # Create unified condition mapping across all pre-filtered sessions
    if verbose:
        print(f"\nCreating unified condition mapping from {len(all_trial_data)} pre-filtered trials...")
    
    unified_condition_mapping = create_unified_condition_mapping(all_trial_data, verbose=verbose)
    
    # Apply unified condition mapping to each session
    if verbose:
        print(f"\nApplying unified conditions to each session...")
    
    final_sessions = {}
    for session_name, (spikes, behavior, trial_data) in all_sessions.items():
        # Apply unified condition mapping
        unified_condition_ids = apply_unified_condition_mapping(trial_data, unified_condition_mapping)
        final_sessions[session_name] = (spikes, behavior, unified_condition_ids, trial_data)
        
        if verbose:
            unique_conds = len(np.unique(unified_condition_ids))
            print(f"  {session_name}: {unique_conds} unique conditions from unified mapping")
    
    if merge_sessions and len(final_sessions) > 1:
        if verbose:
            print(f"\nMerging {len(final_sessions)} sessions with unified conditions...")
        merged_data = merge_session_data_unified(final_sessions, verbose=verbose)
        
        if verbose:
            print(f"\nApplying global neuron filtering on merged data...")
        
        # Apply global neuron filtering on the merged data
        merged_sessions = {"merged": merged_data}
        filtered_sessions = apply_global_neuron_filtering(merged_sessions, verbose=verbose)
        final_sessions["merged"] = filtered_sessions["merged"]
    
    all_sessions = final_sessions
    
    # Save data if output directory specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for session_name, session_data in all_sessions.items():
            if len(session_data) == 4:  # Single session with trial_data
                spikes, behavior, condition_ids, trial_data = session_data
            else:  # Merged session without trial_data
                spikes, behavior, condition_ids = session_data
            
            np.savez(
                output_dir / f"{session_name}_mint_data.npz",
                spikes=spikes,
                behavior=behavior,
                condition_ids=condition_ids
            )
        
        if verbose:
            print(f"\nSaved processed data to {output_dir}")
    
    return all_sessions


def process_single_nwb(nwb_path: Union[str, Path], verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    """Process a single NWB file to extract MINT-compatible data.
    
    Args:
        nwb_path: Path to NWB file
        verbose: Whether to print progress
        
    Returns:
        Tuple of (spikes, behavior, condition_ids, trial_data)
        - spikes: (n_trials, n_neurons, T)
        - behavior: (n_trials, 4, T) [xpos, ypos, xvel, yvel]  
        - condition_ids: (n_trials,) integer labels
        - trial_data: List of trial dictionaries
    """
    with NWBHDF5IO(str(nwb_path), 'r') as io:
        nwbfile = io.read()
        
        # Extract all epochs (trials)
        epochs_df = nwbfile.epochs.to_dataframe()
        
        if verbose:
            print(f"    Found {len(epochs_df)} epochs")
            print(f"    Epoch types: {epochs_df['epoch_type'].unique()}")
            print(f"    Tasks: {epochs_df['task'].unique()}")
        
        # Extract data for each trial
        trial_data = []
        skipped_counts = {'non_movement': 0, 'incompatible': 0, 'extraction_error': 0}
        
        for idx, epoch in epochs_df.iterrows():
            # Check MINT compatibility before extracting data
            from trial_filtering import should_discard_trial
            
            # Create minimal trial info for filtering check
            epoch_type = epoch.get('epoch_type', '')
            task = epoch.get('task', '')
            
            # Quick compatibility check based on epoch type
            non_movement_epochs = ['InterTrial', 'FailSafe', 'Calibration', 'OrthoCalibration', 'Idle']
            if any(epoch_name in epoch_type for epoch_name in non_movement_epochs):
                skipped_counts['non_movement'] += 1
                continue
            
            try:
                trial_info = extract_trial_data(nwbfile, epoch, verbose=verbose)
                if trial_info is not None:
                    # Final compatibility check with full trial data
                    should_discard, reason = should_discard_trial(trial_info)
                    if should_discard:
                        skipped_counts['incompatible'] += 1
                        continue
                    trial_data.append(trial_info)
            except Exception as e:
                skipped_counts['extraction_error'] += 1
                continue
        
        if not trial_data:
            raise ValueError("No valid trials extracted")
        
        if verbose:
            total_skipped = sum(skipped_counts.values())
            print(f"    Extracted {len(trial_data)} MINT-compatible trials, skipped {total_skipped} incompatible trials")
        
        # Convert to arrays
        spikes, behavior, condition_ids = format_trial_arrays(trial_data, verbose=verbose)
        
        # Apply basic data cleaning (no neuron removal - that's done globally later)
        from data_cleaning import clean_data_for_mint
        spikes, behavior, condition_ids, trial_data = clean_data_for_mint(
            spikes, behavior, condition_ids, trial_data, verbose=verbose
        )
        
        return spikes, behavior, condition_ids, trial_data


def extract_trial_data(nwbfile, epoch: pd.Series, verbose: bool = False) -> Optional[Dict]:
    """Extract spike and behavior data for a single trial (epoch).
    
    Args:
        nwbfile: NWB file object
        epoch: Single epoch row from epochs table
        verbose: Whether to print debug info
        
    Returns:
        Dictionary with trial data or None if extraction failed
    """
    start_time = epoch['start_time']
    stop_time = epoch['stop_time']
    duration = stop_time - start_time
    epoch_type = epoch['epoch_type']
    task = epoch['task']
    
    # Find timeseries for this epoch
    spike_data = None
    position_data = None
    velocity_data = None
    target_data = None
    
    # Look for data in processing modules
    # First check ecephys module for spike data
    if 'ecephys' in nwbfile.processing:
        ecephys_module = nwbfile.processing['ecephys']
        for ts_name, ts_obj in ecephys_module.containers.items():
            if 'BinnedSpikes' in ts_name and task in ts_name:
                # Extract data for this epoch's time range
                spike_data = extract_timeseries_data(ts_obj, start_time, duration)
                break
    
    # Then check behavior module for behavioral data
    if 'behavior' in nwbfile.processing:
        behavior_module = nwbfile.processing['behavior']
        
        # Look for MousePosition timeseries (EXCLUDE ClickState - not appropriate for behavior reconstruction)
        for ts_name, ts_obj in behavior_module.containers.items():
            if 'MousePosition' in ts_name and task in ts_name and 'ClickState' not in ts_name:
                position_data = extract_timeseries_data(ts_obj, start_time, duration)
                break
        
        # Look for MouseVelocity timeseries (EXCLUDE ClickState)
        for ts_name, ts_obj in behavior_module.containers.items():
            if 'MouseVelocity' in ts_name and task in ts_name and 'ClickState' not in ts_name:
                velocity_data = extract_timeseries_data(ts_obj, start_time, duration)
                break
        
        # Look for TargetPosition timeseries (optional, EXCLUDE ClickState)
        for ts_name, ts_obj in behavior_module.containers.items():
            if 'TargetPosition' in ts_name and task in ts_name and 'ClickState' not in ts_name:
                target_data = extract_timeseries_data(ts_obj, start_time, duration)
                break
    
    # Check if we have required data
    if spike_data is None:
        if verbose:
            print(f"      No spike data found for {task} epoch")
        return None
    
    if position_data is None or velocity_data is None:
        if verbose:
            print(f"      No position/velocity data found for {task} epoch")
        return None
    
    # Get target position for condition clustering
    target_pos = None
    if target_data is not None and len(target_data) > 0:
        # Use mean target position across the trial
        if target_data.ndim > 1:
            target_pos = np.mean(target_data, axis=0)
        else:
            target_pos = np.mean(target_data)
    
    return {
        'spikes': spike_data,  # (n_neurons, T)
        'position': position_data,  # (T, 2)
        'velocity': velocity_data,  # (T, 2)
        'target': target_pos,  # (2,) or None
        'task': task,
        'epoch_type': epoch_type,
        'duration': duration
    }


def extract_timeseries_data(timeseries_obj, start_time: float, duration: float) -> np.ndarray:
    """Extract data from a timeseries object for a specific time window.
    
    Args:
        timeseries_obj: NWB timeseries object
        start_time: Start time in seconds
        duration: Duration in seconds
        
    Returns:
        Extracted data array
    """
    # Get sampling rate
    rate = timeseries_obj.rate
    
    # Convert times to indices
    start_idx = int(start_time * rate)
    end_idx = int((start_time + duration) * rate)
    
    # Extract data - handle both numpy arrays and h5py datasets
    if hasattr(timeseries_obj.data, 'shape'):
        # For h5py datasets or numpy arrays
        data = timeseries_obj.data[start_idx:end_idx]
        # For BinnedSpikes, transpose to get (n_neurons, T)
        if 'BinnedSpikes' in timeseries_obj.name and data.ndim == 2:
            data = data.T  # Convert from (T, n_neurons) to (n_neurons, T)
    else:
        # Fallback for other data types
        data = np.array(timeseries_obj.data)[start_idx:end_idx]
    
    return data


def format_trial_arrays(trial_data: List[Dict], verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Format extracted trial data into MINT-compatible arrays.
    
    Args:
        trial_data: List of trial dictionaries
        verbose: Whether to print progress
        
    Returns:
        Tuple of (spikes, behavior, condition_ids)
    """
    n_trials = len(trial_data)
    
    # Determine dimensions
    n_neurons = trial_data[0]['spikes'].shape[0]
    max_T = max(trial['spikes'].shape[1] for trial in trial_data)
    
    if verbose:
        print(f"    Formatting arrays: {n_trials} trials, {n_neurons} neurons, {max_T} time bins")
    
    # Initialize arrays
    spikes = np.zeros((n_trials, n_neurons, max_T))
    behavior = np.zeros((n_trials, 4, max_T))
    
    # Fill arrays with padding
    for i, trial in enumerate(trial_data):
        trial_T = trial['spikes'].shape[1]
        
        # Spikes: (n_neurons, T)
        spikes[i, :, :trial_T] = trial['spikes']
        
        # Behavior: combine position and velocity
        pos = trial['position']  # (T, 2)
        vel = trial['velocity']  # (T, 2)
        
        # Ensure consistent length
        min_len = min(pos.shape[0], vel.shape[0], trial_T)
        
        behavior[i, 0, :min_len] = pos[:min_len, 0]  # x position
        behavior[i, 1, :min_len] = pos[:min_len, 1]  # y position
        behavior[i, 2, :min_len] = vel[:min_len, 0]  # x velocity
        behavior[i, 3, :min_len] = vel[:min_len, 1]  # y velocity
    
    # Generate condition IDs
    condition_ids = generate_condition_ids(trial_data, verbose=verbose)
    
    return spikes, behavior, condition_ids


def discretize_target_position(target: Union[List, Tuple, np.ndarray]) -> Tuple[float, float]:
    """Discretize target position following MINT paper guidelines.
    
    Uses coarse spatial binning to avoid over-fragmentation of conditions.
    Based on MINT paper's approach of using angular sectors or grid centers.
    
    Args:
        target: Target position [x, y]
        
    Returns:
        Discretized target position as tuple
    """
    target = np.array(target[:2])  # Ensure we have x, y
    
    # Method 1: Coarse grid binning (0.3 unit bins)
    # This creates fewer, more stable conditions
    bin_size = 0.3
    discretized = np.round(target / bin_size) * bin_size
    
    return tuple(discretized)


def compute_adaptive_target_precision(all_trial_data: List[Dict], verbose: bool = True) -> float:
    """Compute adaptive precision for target rounding based on data distribution.
    
    Args:
        all_trial_data: List of all trial dictionaries from all sessions
        verbose: Whether to print progress
        
    Returns:
        Adaptive precision value for rounding
    """
    # Collect all target positions
    all_targets = []
    for trial in all_trial_data:
        if trial['target'] is not None:
            target = np.array(trial['target'])
            if target.ndim == 0:  # scalar
                all_targets.append([float(target), 0.0])
            elif len(target) == 1:  # 1D
                all_targets.append([float(target[0]), 0.0])
            else:  # 2D or higher
                all_targets.append([float(target[0]), float(target[1])])
    
    if not all_targets:
        if verbose:
            print(f"    No targets found, using default precision 0.1")
        return 0.1
    
    all_targets = np.array(all_targets)
    
    # Compute standard deviation for each dimension
    target_std = np.std(all_targets, axis=0)
    target_range = np.ptp(all_targets, axis=0)  # peak-to-peak (max - min)
    
    # Adaptive precision: aim for ~5-10 clusters per standard deviation
    # but ensure reasonable bounds
    target_clusters_per_std = 7  # Target number of clusters per std dev
    precision_from_std = target_std.mean() / target_clusters_per_std
    
    # Alternative: base on range to get ~10-20 total clusters per dimension
    target_clusters_per_range = 15
    precision_from_range = target_range.mean() / target_clusters_per_range
    
    # Use the more conservative (larger) precision to avoid over-clustering
    adaptive_precision = max(precision_from_std, precision_from_range)
    
    # Enforce reasonable bounds
    min_precision = 0.025  # Don't go below 2.5cm precision
    max_precision = 0.5    # Don't go above 50cm precision
    adaptive_precision = max(min_precision, min(max_precision, adaptive_precision))
    
    # Round to nice values
    nice_precisions = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    adaptive_precision = min(nice_precisions, key=lambda x: abs(x - adaptive_precision))
    
    if verbose:
        print(f"    Target distribution analysis:")
        print(f"      Targets found: {len(all_targets)}")
        print(f"      Target range: x=[{all_targets[:, 0].min():.2f}, {all_targets[:, 0].max():.2f}], y=[{all_targets[:, 1].min():.2f}, {all_targets[:, 1].max():.2f}]")
        print(f"      Target std: x={target_std[0]:.3f}, y={target_std[1]:.3f}")
        print(f"      Adaptive precision: {adaptive_precision}")
        
        # Show how many clusters this will create
        x_clusters = len(np.unique(np.round(all_targets[:, 0] / adaptive_precision)))
        y_clusters = len(np.unique(np.round(all_targets[:, 1] / adaptive_precision)))
        print(f"      Expected clusters: x={x_clusters}, y={y_clusters}")
    
    return adaptive_precision


def create_unified_condition_mapping(all_trial_data: List[Dict], verbose: bool = True) -> Dict:
    """Create unified condition mapping across all trials from all sessions.
    
    Args:
        all_trial_data: List of all trial dictionaries from all sessions
        verbose: Whether to print progress
        
    Returns:
        Dictionary with unified condition mapping
    """
    # Note: Using fixed coarse discretization per MINT paper guidelines
    
    # Create condition keys for all trials
    condition_keys = []
    
    for trial in all_trial_data:
        task = trial['task']
        epoch_type = trial['epoch_type']
        target = trial['target']
        
        # Discretize target position using MINT paper guidelines
        if target is not None:
            rounded_target = discretize_target_position(target)
        else:
            rounded_target = "NO_TARGET"
        
        condition_key = (task, epoch_type, rounded_target)
        condition_keys.append(condition_key)
    
    # Map unique keys to integer IDs
    unique_keys = list(set(condition_keys))
    
    # Sort for consistent ordering across runs (handle different target types properly)
    def sort_key(condition_tuple):
        task, epoch_type, target = condition_tuple
        # Convert different target types to sortable format
        if target == "NO_TARGET":
            target_sort = (0,)  # NO_TARGET conditions come first
        elif isinstance(target, tuple):
            target_sort = (1,) + target  # Real and pseudo-target tuples
        else:
            target_sort = (2, str(target))  # Any other target type comes last
        return (task, epoch_type, target_sort)
    
    unique_keys.sort(key=sort_key)
    key_to_id = {key: i for i, key in enumerate(unique_keys)}
    
    if verbose:
        print(f"    Created unified mapping with {len(unique_keys)} unique conditions:")
        for i, key in enumerate(unique_keys[:20]):  # Show first 20 to avoid spam
            count = condition_keys.count(key)
            print(f"      Condition {i}: {key} ({count} trials across all sessions)")
        if len(unique_keys) > 20:
            print(f"      ... and {len(unique_keys) - 20} more conditions")
    
    return {
        'key_to_id': key_to_id,
        'unique_keys': unique_keys,
        'total_trials': len(all_trial_data),
        'discretization_method': 'coarse_grid_0.3'
    }


def apply_unified_condition_mapping(trial_data: List[Dict], unified_mapping: Dict) -> np.ndarray:
    """Apply unified condition mapping to a session's trial data.
    
    Args:
        trial_data: List of trial dictionaries for this session
        unified_mapping: Unified condition mapping from create_unified_condition_mapping
        
    Returns:
        Array of condition IDs (n_trials,)
    """
    key_to_id = unified_mapping['key_to_id']
    # Using coarse discretization method from mapping
    condition_ids = []
    
    for trial in trial_data:
        task = trial['task']
        epoch_type = trial['epoch_type']
        target = trial['target']
        
        # Discretize target position using MINT paper guidelines
        if target is not None:
            rounded_target = discretize_target_position(target)
        else:
            rounded_target = "NO_TARGET"
        
        condition_key = (task, epoch_type, rounded_target)
        condition_ids.append(key_to_id[condition_key])
    
    return np.array(condition_ids)


def generate_condition_ids(trial_data: List[Dict], verbose: bool = True) -> np.ndarray:
    """Generate condition IDs based on task, epoch type, and target position.
    
    NOTE: This function is kept for backward compatibility but should not be used 
    when processing multiple sessions. Use create_unified_condition_mapping instead.
    
    Args:
        trial_data: List of trial dictionaries
        verbose: Whether to print progress
        
    Returns:
        Array of condition IDs (n_trials,)
    """
    # Create condition keys
    condition_keys = []
    
    for trial in trial_data:
        task = trial['task']
        epoch_type = trial['epoch_type']
        target = trial['target']
        
        # Discretize target position following MINT paper guidelines
        if target is not None:
            rounded_target = discretize_target_position(target)
        else:
            rounded_target = None
        
        condition_key = (task, epoch_type, rounded_target)
        condition_keys.append(condition_key)
    
    # Map unique keys to integer IDs
    unique_keys = list(set(condition_keys))
    key_to_id = {key: i for i, key in enumerate(unique_keys)}
    
    condition_ids = np.array([key_to_id[key] for key in condition_keys])
    
    if verbose:
        print(f"    Generated {len(unique_keys)} unique conditions:")
        for i, key in enumerate(unique_keys):
            count = np.sum(condition_ids == i)
            print(f"      Condition {i}: {key} ({count} trials)")
    
    return condition_ids


def apply_global_neuron_filtering(sessions: Dict[str, Tuple], 
                                 firing_rate_threshold: float = 1.0,
                                 verbose: bool = True) -> Dict[str, Tuple]:
    """Apply global neuron filtering on merged session data.
    
    Calculates firing rates across the merged data and removes neurons
    below the threshold.
    
    Args:
        sessions: Dictionary with single "merged" session: (spikes, behavior, condition_ids)
        firing_rate_threshold: Minimum firing rate in Hz (default 1.0 as per MINT paper)
        verbose: Whether to print progress
        
    Returns:
        Updated sessions dictionary with filtered neurons
    """
    if len(sessions) != 1 or "merged" not in sessions:
        raise ValueError("apply_global_neuron_filtering expects a single 'merged' session")
    
    spikes, behavior, condition_ids = sessions["merged"]
    
    if verbose:
        print(f"  Calculating global firing rates on merged data...")
        print(f"  Data shape: {spikes.shape[0]} trials, {spikes.shape[1]} neurons, {spikes.shape[2]} time bins")
    
    # Calculate mean firing rate per neuron across all trials
    # Convert from spikes/bin to spikes/second (assuming 20ms bins)
    bin_size_sec = 0.02  # 20ms bins
    mean_rates_per_neuron = np.mean(spikes, axis=(0, 2)) / bin_size_sec
    
    # Find neurons above threshold
    active_neuron_mask = mean_rates_per_neuron >= firing_rate_threshold
    active_neurons = np.where(active_neuron_mask)[0]
    
    if verbose:
        print(f"  Global firing rates: min={mean_rates_per_neuron.min():.3f}, max={mean_rates_per_neuron.max():.3f}, mean={mean_rates_per_neuron.mean():.3f} Hz")
        print(f"  Keeping {len(active_neurons)}/{spikes.shape[1]} neurons above {firing_rate_threshold} Hz threshold")
        print(f"  Removed neurons: {np.where(~active_neuron_mask)[0].tolist()}")
    
    # Apply neuron filtering
    filtered_spikes = spikes[:, active_neuron_mask, :]
    
    # For merged data, we skip the full clean_data_for_mint since we don't have trial_data structure
    # Just apply the essential cleaning steps manually
    from data_cleaning import remove_static_trials, smooth_sparse_data
    
    # Create dummy trial_data for the number of trials we have
    dummy_trial_data = [{"trial_id": i} for i in range(filtered_spikes.shape[0])]
    
    # Remove static trials
    cleaned_spikes, cleaned_behavior, cleaned_condition_ids, _ = remove_static_trials(
        filtered_spikes, behavior, condition_ids, dummy_trial_data, verbose=False
    )
    
    # Light smoothing for very sparse data
    sparsity = (cleaned_spikes == 0).sum() / cleaned_spikes.size * 100
    if sparsity > 90:  # Very sparse data
        cleaned_spikes = smooth_sparse_data(cleaned_spikes, sigma=1.5, verbose=False)
    
    # Add small regularization to prevent exact zeros
    cleaned_spikes = cleaned_spikes + 1e-8
    cleaned_behavior = cleaned_behavior + 1e-10
    
    if verbose:
        final_sparsity = (cleaned_spikes < 1e-6).sum() / cleaned_spikes.size * 100
        print(f"  Final: {cleaned_spikes.shape[0]} trials, {cleaned_spikes.shape[1]} neurons")
        print(f"  Final sparsity: {final_sparsity:.1f}%")
        print(f"  Neural rate range: [{cleaned_spikes.min():.6f}, {cleaned_spikes.max():.6f}]")
    
    return {"merged": (cleaned_spikes, cleaned_behavior, cleaned_condition_ids)}


def merge_session_data_unified(sessions: Dict[str, Tuple], 
                              verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge data from multiple sessions with unified condition IDs.
    
    Args:
        sessions: Dictionary mapping session names to (spikes, behavior, condition_ids, trial_data)
        verbose: Whether to print progress
        
    Returns:
        Merged (spikes, behavior, condition_ids) tuple
    """
    if not sessions:
        raise ValueError("No sessions to merge")
    
    # Skip merged session if it already exists
    session_data = {k: v for k, v in sessions.items() if k != "merged"}
    
    if len(session_data) == 1:
        # Return first 3 elements (spikes, behavior, condition_ids)
        return list(session_data.values())[0][:3]
    
    # Check dimensions
    first_session = list(session_data.values())[0]
    n_neurons = first_session[0].shape[1]
    max_T = max(data[0].shape[2] for data in session_data.values())
    
    # Collect all data
    all_spikes = []
    all_behavior = []
    all_condition_ids = []
    
    for session_name, session_tuple in session_data.items():
        spikes, behavior, condition_ids, trial_data = session_tuple
        
        # All sessions should have same number of neurons after global filtering
        assert spikes.shape[1] == n_neurons, f"Session {session_name} has {spikes.shape[1]} neurons, expected {n_neurons}"
        
        # Pad to max_T if needed
        if spikes.shape[2] < max_T:
            padded_spikes = np.zeros((spikes.shape[0], spikes.shape[1], max_T))
            padded_behavior = np.zeros((behavior.shape[0], behavior.shape[1], max_T))
            
            padded_spikes[:, :, :spikes.shape[2]] = spikes
            padded_behavior[:, :, :behavior.shape[2]] = behavior
            
            spikes = padded_spikes
            behavior = padded_behavior
        
        # No need to offset condition IDs - they're already unified!
        all_spikes.append(spikes)
        all_behavior.append(behavior)
        all_condition_ids.append(condition_ids)
        
        if verbose:
            unique_conds = len(np.unique(condition_ids))
            print(f"    Session {session_name}: {spikes.shape[0]} trials, {unique_conds} unique conditions (unified)")
    
    # Concatenate
    merged_spikes = np.concatenate(all_spikes, axis=0)
    merged_behavior = np.concatenate(all_behavior, axis=0)
    merged_condition_ids = np.concatenate(all_condition_ids)
    
    if verbose:
        print(f"    Merged result: {merged_spikes.shape[0]} trials, {len(np.unique(merged_condition_ids))} unified conditions")
    
    return merged_spikes, merged_behavior, merged_condition_ids


def merge_session_data(sessions: Dict[str, Tuple], 
                      verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legacy merge function - kept for backward compatibility.
    
    NOTE: This function uses condition ID offsetting. Use merge_session_data_unified 
    for proper unified condition handling.
    """
    if not sessions:
        raise ValueError("No sessions to merge")
    
    # Skip merged session if it already exists
    session_data = {k: v for k, v in sessions.items() if k != "merged"}
    
    if len(session_data) == 1:
        return list(session_data.values())[0]
    
    # Check dimensions
    first_session = list(session_data.values())[0]
    if len(first_session) == 4:
        n_neurons = first_session[0].shape[1]
        max_T = max(data[0].shape[2] for data in session_data.values())
    else:
        n_neurons = first_session[0].shape[1]
        max_T = max(data[0].shape[2] for data in session_data.values())
    
    # Collect all data
    all_spikes = []
    all_behavior = []
    all_condition_ids = []
    
    condition_offset = 0
    
    for session_name, session_tuple in session_data.items():
        # Handle both 3-tuple (merged) and 4-tuple (single session) formats
        if len(session_tuple) == 4:
            spikes, behavior, condition_ids, _ = session_tuple
        else:
            spikes, behavior, condition_ids = session_tuple
        # Pad to max_T if needed
        if spikes.shape[2] < max_T:
            padded_spikes = np.zeros((spikes.shape[0], spikes.shape[1], max_T))
            padded_behavior = np.zeros((behavior.shape[0], behavior.shape[1], max_T))
            
            padded_spikes[:, :, :spikes.shape[2]] = spikes
            padded_behavior[:, :, :behavior.shape[2]] = behavior
            
            spikes = padded_spikes
            behavior = padded_behavior
        
        # Offset condition IDs to avoid conflicts
        offset_condition_ids = condition_ids + condition_offset
        condition_offset = offset_condition_ids.max() + 1
        
        all_spikes.append(spikes)
        all_behavior.append(behavior)
        all_condition_ids.append(offset_condition_ids)
        
        if verbose:
            print(f"    Session {session_name}: {spikes.shape[0]} trials, conditions {condition_ids.min()}-{condition_ids.max()} -> {offset_condition_ids.min()}-{offset_condition_ids.max()}")
    
    # Concatenate
    merged_spikes = np.concatenate(all_spikes, axis=0)
    merged_behavior = np.concatenate(all_behavior, axis=0)
    merged_condition_ids = np.concatenate(all_condition_ids)
    
    if verbose:
        print(f"    Merged result: {merged_spikes.shape[0]} trials, {len(np.unique(merged_condition_ids))} conditions")
    
    return merged_spikes, merged_behavior, merged_condition_ids


def split_train_test(spikes: np.ndarray, 
                     behavior: np.ndarray, 
                     condition_ids: np.ndarray,
                     test_fraction: float = 0.2,
                     verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train/test sets, stratified by condition.
    
    Args:
        spikes: Neural spike data [trials, neurons, time]
        behavior: Behavioral data [trials, behavior_dims, time]
        condition_ids: Condition IDs [trials]
        test_fraction: Fraction of data to use for testing
        verbose: Whether to print progress
        
    Returns:
        Tuple of (train_spikes, train_behavior, train_conds, test_spikes, test_behavior, test_conds)
    """
    unique_conds = np.unique(condition_ids)
    train_indices = []
    test_indices = []
    
    # Split each condition separately to maintain balance
    for cond in unique_conds:
        cond_mask = condition_ids == cond
        cond_indices = np.where(cond_mask)[0]
        n_cond = len(cond_indices)
        n_test_cond = max(1, int(n_cond * test_fraction))  # At least 1 test trial per condition
        
        # Shuffle within condition
        np.random.shuffle(cond_indices)
        test_indices.extend(cond_indices[:n_test_cond])
        train_indices.extend(cond_indices[n_test_cond:])
    
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    
    # Split data
    train_spikes = spikes[train_indices]
    train_behavior = behavior[train_indices]
    train_conds = condition_ids[train_indices]
    
    test_spikes = spikes[test_indices]
    test_behavior = behavior[test_indices]
    test_conds = condition_ids[test_indices]
    
    if verbose:
        print(f"Train/test split:")
        print(f"  Train trials: {len(train_indices)}")
        print(f"  Test trials: {len(test_indices)}")
        
        # Check trials per condition in training set
        for cond in unique_conds:
            n_train_cond = np.sum(train_conds == cond)
            n_test_cond = np.sum(test_conds == cond)
            print(f"  Condition {cond}: {n_train_cond} train, {n_test_cond} test")
    
    return train_spikes, train_behavior, train_conds, test_spikes, test_behavior, test_conds


def create_mint_settings(train_conds: np.ndarray, 
                        train_spikes: np.ndarray,
                        train_behavior: np.ndarray,
                        trial_data: List[Dict],
                        output_path: Path,
                        use_movement_alignment: bool = True):
    """Create MINTSettings dynamically optimized for the data.
    
    Args:
        train_conds: Training condition IDs
        train_spikes: Training spike data [n_trials, n_neurons, T]
        train_behavior: Training behavior data [n_trials, 4, T]
        trial_data: Original trial data with durations and sampling info
        output_path: Output directory path
        use_movement_alignment: Whether to use movement-based alignment
        
    Returns:
        Configured MINTSettings object
    """
    from brn.mint.decoder import MINTSettings
    
    # Extract timing information from trial data
    durations = np.array([trial['duration'] for trial in trial_data])
    
    # Try to extract sampling rate from first trial with spike data
    first_trial = trial_data[0]
    spike_T = first_trial['spikes'].shape[1]
    duration = first_trial['duration']
    estimated_rate = spike_T / duration  # samples per second
    bin_size_ms = 1000 / estimated_rate  # milliseconds per bin
    
    # Round to nearest sensible bin size
    if bin_size_ms <= 25:
        bin_size = 20  # 50 Hz
        sampling_period = 0.02
    elif bin_size_ms <= 35:
        bin_size = 30  # ~33 Hz
        sampling_period = 0.03
    elif bin_size_ms <= 45:
        bin_size = 40  # 25 Hz
        sampling_period = 0.04
    else:
        bin_size = 50  # 20 Hz
        sampling_period = 0.05
    
    # Compute alignment windows
    if use_movement_alignment:
        # Use movement-based alignment
        print("    Computing movement-based alignment windows...")
        movement_windows = compute_movement_windows(
            trial_data=trial_data,
            sampling_rate=estimated_rate,
            verbose=True
        )
        trial_alignment = movement_windows['trial_alignment']
        trajectories_alignment = movement_windows['trajectories_alignment']
        test_alignment = movement_windows['test_alignment']
        
    else:
        # Use fixed alignment windows based on data length
        T = train_spikes.shape[2]  # Number of time bins
        total_duration_ms = T * bin_size
        
        # Trial alignment: use full data range
        trial_start_ms = 0
        trial_end_ms = total_duration_ms
        trial_alignment = np.arange(trial_start_ms, trial_end_ms, bin_size)
        
        # Trajectories alignment: use central 80% of data
        traj_margin = int(0.1 * T)  # 10% margin on each side
        traj_start_ms = traj_margin * bin_size
        traj_end_ms = (T - traj_margin) * bin_size
        trajectories_alignment = np.arange(traj_start_ms, traj_end_ms, bin_size)
        
        # Test alignment: use central 60% of data
        test_margin = int(0.2 * T)  # 20% margin on each side
        test_start_ms = test_margin * bin_size
        test_end_ms = (T - test_margin) * bin_size
        test_alignment = np.arange(test_start_ms, test_end_ms, bin_size)
    
    # Determine appropriate PCA dimensions based on data size
    unique_conds = np.unique(train_conds)
    min_trials_per_cond = min(np.sum(train_conds == cond) for cond in unique_conds)
    
    # Trial dims: constrained by minimum trials per condition
    trial_dims = min(5, max(1, min_trials_per_cond - 2))  # Leave degrees of freedom
    
    # Condition dims: scale with number of conditions but cap reasonably
    condition_dims = min(12, max(2, len(unique_conds) * 2))
    
    # Neural dims: use fraction of neurons but ensure it's reasonable
    n_neurons = train_spikes.shape[1]
    neural_dims = min(20, max(5, n_neurons // 3))
    
    # Observation window: adaptive based on trial alignment length
    # Use ~15-30% of trial length, but cap between 100-500ms
    obs_window_fraction = 0.2
    total_duration_ms = len(trial_alignment) * bin_size
    adaptive_obs_window = int(total_duration_ms * obs_window_fraction)
    observation_window = max(100, min(500, adaptive_obs_window))
    
    # Gaussian sigma: adaptive smoothing based on bin size
    # Use 1.5x bin size for reasonable smoothing
    gaussian_sigma = int(bin_size * 1.5)
    
    # Determine if we should use causal decoding
    # Use causal if trials are long enough for meaningful prediction
    causal = total_duration_ms > 500  # Use causal for longer trials
    
    print(f"    Dynamic MINT Settings computed:")
    print(f"      Estimated sampling: {estimated_rate:.1f} Hz -> {bin_size}ms bins")
    print(f"      Trial duration: {total_duration_ms:.0f}ms ({len(trial_alignment)} bins)")
    print(f"      Trial alignment: {trial_alignment[0]:.0f}-{trial_alignment[-1]:.0f}ms ({len(trial_alignment)} bins)")
    print(f"      Trajectories alignment: {trajectories_alignment[0]:.0f}-{trajectories_alignment[-1]:.0f}ms ({len(trajectories_alignment)} bins)")
    print(f"      Test alignment: {test_alignment[0]:.0f}-{test_alignment[-1]:.0f}ms ({len(test_alignment)} bins)")
    print(f"      Observation window: {observation_window}ms")
    print(f"      PCA dims - Neural: {neural_dims}, Condition: {condition_dims}, Trial: {trial_dims}")
    print(f"      Causal decoding: {causal}")
    
    # Configure MINT settings for template mode
    # Note: preprocessing parameters are not needed since we use template mode
    settings = MINTSettings(
        # Core settings
        task="generalized_click",
        data_path=str(output_path / "data"),
        results_path=str(output_path / "results"),
        
        # Timing settings - computed from data
        bin_size=bin_size,
        sampling_period=sampling_period,
        
        # Decoding settings - adaptive
        observation_window=observation_window,
        causal=causal,
        
        # Test alignment - required for decoding
        test_alignment=test_alignment,
        
        # Regularization
        min_lambda=1e-8,
        
        # Optional preprocessing parameters (not used in template mode but saved for reference)
        trial_alignment=trial_alignment,
        trajectories_alignment=trajectories_alignment,
        gaussian_sigma=gaussian_sigma,
        neural_dims=neural_dims,
        condition_dims=condition_dims,
        trial_dims=trial_dims,
    )
    
    return settings


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R-squared score.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        R-squared score
    """
    # Flatten arrays
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    
    # Remove any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    # Calculate R2
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0