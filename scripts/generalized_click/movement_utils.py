"""Movement detection and temporal alignment utilities.

Detects movement onset/offset from behavioral data and computes
data-driven alignment windows for MINT training.
"""

import numpy as np
from typing import List, Tuple, Dict
from scipy.ndimage import gaussian_filter1d


def detect_movement_onset_offset(velocity: np.ndarray, 
                                threshold_percentile: float = 20,
                                min_duration_ms: float = 100,
                                sampling_rate: float = 50.0,
                                smooth_sigma_ms: float = 50) -> Tuple[int, int, Dict]:
    """Detect movement onset and offset from velocity data.
    
    Args:
        velocity: Velocity data (2, T) for x and y components
        threshold_percentile: Percentile of velocity magnitude to use as threshold
        min_duration_ms: Minimum duration of movement in milliseconds
        sampling_rate: Sampling rate in Hz
        smooth_sigma_ms: Gaussian smoothing sigma in milliseconds
        
    Returns:
        Tuple of (onset_idx, offset_idx, detection_info) 
        detection_info contains metadata about the detection process
    """
    # Initialize detection info
    detection_info = {
        'success': True,
        'fallback_reason': None,
        'vel_magnitude_stats': {},
        'threshold_used': None,
        'movement_detected': True
    }
    
    try:
        # Compute velocity magnitude
        vel_magnitude = np.sqrt(velocity[0]**2 + velocity[1]**2)
        
        # Store velocity stats
        detection_info['vel_magnitude_stats'] = {
            'mean': float(np.mean(vel_magnitude)),
            'std': float(np.std(vel_magnitude)),
            'max': float(np.max(vel_magnitude)),
            'min': float(np.min(vel_magnitude))
        }
        
        # Smooth velocity
        sigma_samples = smooth_sigma_ms * sampling_rate / 1000
        vel_smooth = gaussian_filter1d(vel_magnitude, sigma=sigma_samples)
        
        # Compute threshold
        threshold = np.percentile(vel_smooth, threshold_percentile)
        detection_info['threshold_used'] = float(threshold)
        
        # Find movement periods
        is_moving = vel_smooth > threshold
        
        # Minimum duration in samples
        min_samples = int(min_duration_ms * sampling_rate / 1000)
        
        # Find onset - first sustained movement
        onset_idx = 0
        onset_found = False
        for i in range(len(is_moving) - min_samples):
            if np.all(is_moving[i:i+min_samples]):
                onset_idx = i
                onset_found = True
                break
        
        # Find offset - last sustained movement
        offset_idx = len(is_moving) - 1
        offset_found = False
        for i in range(len(is_moving) - min_samples, -1, -1):
            if np.all(is_moving[i:i+min_samples]):
                offset_idx = i + min_samples - 1
                offset_found = True
                break
        
        # Check for detection failures
        if not onset_found or not offset_found:
            detection_info['success'] = False
            detection_info['fallback_reason'] = 'no_sustained_movement_found'
            detection_info['movement_detected'] = False
            onset_idx = 0
            offset_idx = len(is_moving) - 1
        elif onset_idx >= offset_idx:
            detection_info['success'] = False
            detection_info['fallback_reason'] = 'onset_after_offset'
            onset_idx = 0
            offset_idx = len(is_moving) - 1
        elif (offset_idx - onset_idx) < min_samples:
            detection_info['success'] = False
            detection_info['fallback_reason'] = 'movement_too_short'
            onset_idx = 0
            offset_idx = len(is_moving) - 1
            
    except Exception as e:
        # Complete fallback on any error
        detection_info['success'] = False
        detection_info['fallback_reason'] = f'exception: {str(e)}'
        detection_info['movement_detected'] = False
        onset_idx = 0
        offset_idx = len(velocity[0]) - 1
    
    return onset_idx, offset_idx, detection_info


def compute_movement_windows(trial_data: List[Dict], 
                           sampling_rate: float = 50.0,
                           verbose: bool = True) -> Dict[str, np.ndarray]:
    """Compute data-driven alignment windows based on movement patterns.
    
    Args:
        trial_data: List of trial dictionaries with velocity data
        sampling_rate: Sampling rate in Hz
        verbose: Whether to print progress
        
    Returns:
        Dictionary with alignment arrays for trial, trajectories, and test windows
    """
    all_onsets = []
    all_offsets = []
    all_durations = []
    all_trial_lengths = []
    detection_failures = []
    
    # Detect movement for each trial
    for i, trial in enumerate(trial_data):
        velocity = trial['velocity'].T  # Convert to (2, T)
        trial_length = velocity.shape[1]
        all_trial_lengths.append(trial_length)
        
        try:
            onset_idx, offset_idx, detection_info = detect_movement_onset_offset(
                velocity, 
                sampling_rate=sampling_rate
            )
            
            # Log detection failures
            if not detection_info['success']:
                detection_failures.append({
                    'trial_idx': i,
                    'reason': detection_info['fallback_reason'],
                    'trial_length': trial_length,
                    'vel_stats': detection_info['vel_magnitude_stats']
                })
            
            all_onsets.append(onset_idx)
            all_offsets.append(offset_idx)
            all_durations.append(offset_idx - onset_idx)
            
        except Exception as e:
            # Use full trial if movement detection fails
            detection_failures.append({
                'trial_idx': i,
                'reason': f'exception: {str(e)}',
                'trial_length': trial_length,
                'vel_stats': {}
            })
            all_onsets.append(0)
            all_offsets.append(trial_length - 1)
            all_durations.append(trial_length - 1)
    
    # Compute statistics
    mean_onset = np.mean(all_onsets)
    mean_offset = np.mean(all_offsets)
    mean_duration = np.mean(all_durations)
    
    # Use median for robustness
    median_onset = np.median(all_onsets)
    median_offset = np.median(all_offsets)
    median_duration = np.median(all_durations)
    
    # Trial length statistics for bounds checking
    min_trial_length = min(all_trial_lengths)
    max_trial_length = max(all_trial_lengths)
    median_trial_length = np.median(all_trial_lengths)
    
    # Log detection failures
    if verbose and detection_failures:
        print(f"\n    Movement detection failures: {len(detection_failures)}/{len(trial_data)} trials")
        failure_reasons = {}
        for failure in detection_failures:
            reason = failure['reason']
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        for reason, count in failure_reasons.items():
            print(f"      {reason}: {count} trials")
    
    if verbose:
        print(f"\n    Movement statistics across {len(trial_data)} trials:")
        print(f"      Onset: mean={mean_onset:.1f}, median={median_onset:.1f} samples")
        print(f"      Offset: mean={mean_offset:.1f}, median={median_offset:.1f} samples")
        print(f"      Duration: mean={mean_duration:.1f}, median={median_duration:.1f} samples")
        print(f"      Trial lengths: min={min_trial_length}, max={max_trial_length}, median={median_trial_length:.1f} samples")
    
    # Define alignment windows based on movement patterns
    # Convert to milliseconds
    ms_per_sample = 1000 / sampling_rate
    
    # Trial alignment: include some pre-movement and post-movement
    pre_movement_ms = 200  # 200ms before movement
    post_movement_ms = 200  # 200ms after movement
    
    trial_start_ms = max(0, (median_onset * ms_per_sample) - pre_movement_ms)
    trial_end_ms = (median_offset * ms_per_sample) + post_movement_ms
    
    # BOUNDS CHECKING: Ensure alignment windows fit within actual trial bounds
    max_trial_duration_ms = max_trial_length * ms_per_sample
    min_trial_duration_ms = min_trial_length * ms_per_sample
    
    # Adjust if trial_end_ms exceeds the shortest trial
    if trial_end_ms > min_trial_duration_ms:
        if verbose:
            print(f"\n    WARNING: Trial alignment end ({trial_end_ms:.0f}ms) exceeds shortest trial ({min_trial_duration_ms:.0f}ms)")
        trial_end_ms = min_trial_duration_ms - ms_per_sample  # Leave one sample margin
        
    # Ensure we have a reasonable minimum window
    if trial_end_ms - trial_start_ms < 400:  # At least 400ms
        if verbose:
            print(f"    WARNING: Trial window too short, using fallback bounds")
        trial_start_ms = 0
        trial_end_ms = min(800, min_trial_duration_ms - ms_per_sample)  # 800ms or trial length
    
    # Round to nearest bin size (20ms)
    bin_size = 20
    trial_start_ms = np.round(trial_start_ms / bin_size) * bin_size
    trial_end_ms = np.round(trial_end_ms / bin_size) * bin_size
    
    # Create alignment arrays
    trial_alignment = np.arange(trial_start_ms, trial_end_ms, bin_size)
    
    # Trajectories alignment: focus on movement period (within trial bounds)
    traj_start_ms = max(trial_start_ms, median_onset * ms_per_sample)
    traj_end_ms = min(trial_end_ms, median_offset * ms_per_sample)
    traj_start_ms = np.round(traj_start_ms / bin_size) * bin_size
    traj_end_ms = np.round(traj_end_ms / bin_size) * bin_size
    
    # Ensure trajectories window is valid
    if traj_end_ms <= traj_start_ms:
        traj_start_ms = trial_start_ms
        traj_end_ms = trial_end_ms
        if verbose:
            print(f"    WARNING: Invalid trajectories window, using full trial window")
    
    trajectories_alignment = np.arange(traj_start_ms, traj_end_ms, bin_size)
    
    # Test alignment: central portion of trajectories (within trajectory bounds)
    test_margin = 0.2  # Use central 60% of movement
    test_duration = traj_end_ms - traj_start_ms
    test_start_ms = traj_start_ms + test_duration * test_margin
    test_end_ms = traj_end_ms - test_duration * test_margin
    test_start_ms = np.round(test_start_ms / bin_size) * bin_size
    test_end_ms = np.round(test_end_ms / bin_size) * bin_size
    
    # Ensure test window is valid
    if test_end_ms <= test_start_ms or test_end_ms - test_start_ms < 200:  # At least 200ms
        test_start_ms = traj_start_ms
        test_end_ms = traj_end_ms
        if verbose:
            print(f"    WARNING: Invalid test window, using full trajectories window")
    
    test_alignment = np.arange(test_start_ms, test_end_ms, bin_size)
    
    # Final validation
    alignment_valid = True
    validation_warnings = []
    
    if len(trial_alignment) == 0:
        validation_warnings.append("Empty trial alignment")
        alignment_valid = False
        # Create minimal fallback alignment
        trial_alignment = np.array([0, 20, 40])  # Minimal 3-bin alignment
    
    if len(trajectories_alignment) == 0:
        validation_warnings.append("Empty trajectories alignment")
        alignment_valid = False
        # Use trial alignment as fallback
        trajectories_alignment = trial_alignment.copy()
    
    if len(test_alignment) == 0:
        validation_warnings.append("Empty test alignment")
        alignment_valid = False
        # Use trajectories alignment as fallback
        test_alignment = trajectories_alignment.copy()
    
    # Check bounds only if we have valid alignments
    if len(trial_alignment) > 0 and trial_alignment[-1] > min_trial_duration_ms:
        validation_warnings.append(f"Trial alignment exceeds shortest trial ({trial_alignment[-1]:.0f}ms > {min_trial_duration_ms:.0f}ms)")
        alignment_valid = False
    
    if verbose:
        print(f"\n    Data-driven alignment windows:")
        print(f"      Trial: {trial_start_ms:.0f}-{trial_end_ms:.0f}ms ({len(trial_alignment)} bins)")
        print(f"      Trajectories: {traj_start_ms:.0f}-{traj_end_ms:.0f}ms ({len(trajectories_alignment)} bins)")
        print(f"      Test: {test_start_ms:.0f}-{test_end_ms:.0f}ms ({len(test_alignment)} bins)")
        
        if validation_warnings:
            print(f"\n    ALIGNMENT VALIDATION WARNINGS:")
            for warning in validation_warnings:
                print(f"      - {warning}")
    
    return {
        'trial_alignment': trial_alignment,
        'trajectories_alignment': trajectories_alignment,
        'test_alignment': test_alignment,
        'movement_stats': {
            'onset_samples': all_onsets,
            'offset_samples': all_offsets,
            'duration_samples': all_durations,
            'trial_lengths': all_trial_lengths,
            'median_onset_ms': median_onset * ms_per_sample,
            'median_offset_ms': median_offset * ms_per_sample,
            'median_duration_ms': median_duration * ms_per_sample,
            'detection_failures': detection_failures,
            'alignment_valid': alignment_valid,
            'validation_warnings': validation_warnings
        }
    }