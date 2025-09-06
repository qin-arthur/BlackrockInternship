"""
MINT preprocessing functions for spike and behavioral data.

This module contains preprocessing functions that can be optionally applied
before fitting the MINT decoder. Different tasks may require different
preprocessing approaches:

- mc_maze, area2_bump: Use standard_preprocessing() for Gaussian smoothing and PCA
- mc_rtt with LFADS: Skip preprocessing as data is already smoothed
"""

import numpy as np
import numpy.typing as npt
from typing import Dict, Tuple
from . import utils


def standard_preprocessing(
    spikes: npt.NDArray,
    behavior: npt.NDArray,
    cond_ids: npt.NDArray,
    trial_alignment: npt.NDArray,
    trajectories_alignment: npt.NDArray,
    gaussian_sigma: int,
    bin_size: int,
    soft_norm: float,
    sampling_period: float,
    trial_dims: int,
    neural_dims: int,
    condition_dims: int,
) -> Tuple[Dict[int, npt.NDArray], Dict[int, npt.NDArray], npt.NDArray]:
    """
    Standard MINT preprocessing for mc_maze and area2_bump tasks.
    
    Performs the following steps:
    1. Smooth spikes with Gaussian filter
    2. Process kinematics (baseline alignment)
    3. Trim data to trajectory alignment window
    4. Group by conditions
    5. Compute trial-averaged behavior
    6. Apply dimensionality reduction to spike data
    
    Parameters
    ----------
    spikes : ndarray, shape (n_trials, n_neurons, n_timepoints)
        Raw spike count data
    behavior : ndarray, shape (n_trials, n_kin_vars, n_timepoints)
        Raw kinematic data  
    cond_ids : ndarray, shape (n_trials,)
        Condition ID for each trial
    trial_alignment : ndarray
        Time points (ms) relative to movement onset for trial data
    trajectories_alignment : ndarray
        Time points (ms) relative to movement onset for trajectory templates
    gaussian_sigma : int
        Gaussian filter sigma (ms)
    bin_size : int
        Bin size for data (ms)
    soft_norm : float
        PCA soft normalization constant
    sampling_period : float
        Sampling period (seconds)
    trial_dims : int
        Number of PCA components for trial-level reduction
    neural_dims : int
        Number of PCA components for neural reduction (or np.nan to skip)
    condition_dims : int
        Number of PCA components for condition-level reduction (or np.nan to skip)
        
    Returns
    -------
    rate_templates : Dict[int, ndarray]
        Trial-averaged, smoothed firing rate templates by condition
    behavior_templates : Dict[int, ndarray]
        Trial-averaged kinematic templates by condition  
    condition_list : ndarray
        Array mapping condition indices to actual condition IDs
    """
    
    # Step 1: Smooth spikes with a Gaussian filter
    spikes_smooth = np.array(
        [utils.gauss_filt(spike, gaussian_sigma, bin_size) for spike in spikes]
    )
    
    # Step 2: Process kinematics (baseline alignment)
    behavior_processed = utils.process_kinematics(behavior, trial_alignment)
    
    # Step 3: Trim smoothed spikes and kinematics in time to match trajectory alignment
    t_mask = np.isin(trial_alignment, trajectories_alignment)
    spikes_smooth = spikes_smooth[:, :, t_mask]
    behavior_processed = behavior_processed[:, :, t_mask]
    
    # Step 4: Reformat into dictionaries grouped by condition
    cond_list = np.unique(cond_ids)
    X = {cond: spikes_smooth[cond_ids == cond] for cond in cond_list}
    Z_sort = {cond: behavior_processed[cond_ids == cond] for cond in cond_list}
    
    # Step 5: Compute trial-averaged kinematics
    behavior_templates = {k: np.mean(v, axis=0) for k, v in Z_sort.items()}
    
    # Step 6: Create smooth firing rate averages using dimensionality reduction
    rate_templates = utils.smooth_average(
        X,
        soft_norm,
        bin_size,
        sampling_period,
        trial_dims,
        neural_dims,
        condition_dims
    )
    
    return rate_templates, behavior_templates, cond_list


def standard_preprocessing_array(
    spikes: npt.NDArray,
    behavior: npt.NDArray,
    cond_ids: npt.NDArray,
    trial_alignment: npt.NDArray,
    trajectories_alignment: npt.NDArray,
    gaussian_sigma: int,
    bin_size: int,
    soft_norm: float,
    sampling_period: float,
    trial_dims: int,
    neural_dims: int,
    condition_dims: int,
    return_dict: bool = False,
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Standard MINT preprocessing that returns arrays suitable for online MINT.
    
    This function performs the same preprocessing as standard_preprocessing()
    but returns data in array format with trial-averaged templates already computed.
    The output can be directly binned and fed to the online MINT decoder.
    
    Parameters
    ----------
    spikes : ndarray, shape (n_trials, n_neurons, n_timepoints)
        Raw spike count data
    behavior : ndarray, shape (n_trials, n_kin_vars, n_timepoints)
        Raw kinematic data  
    cond_ids : ndarray, shape (n_trials,)
        Condition ID for each trial
    trial_alignment : ndarray
        Time points (ms) relative to movement onset for trial data
    trajectories_alignment : ndarray
        Time points (ms) relative to movement onset for trajectory templates
    gaussian_sigma : int
        Gaussian filter sigma (ms)
    bin_size : int
        Bin size for data (ms)
    soft_norm : float
        PCA soft normalization constant
    sampling_period : float
        Sampling period (seconds)
    trial_dims : int
        Number of PCA components for trial-level reduction
    neural_dims : int
        Number of PCA components for neural reduction (or np.nan to skip)
    condition_dims : int
        Number of PCA components for condition-level reduction (or np.nan to skip)
    return_dict : bool
        If True, return dictionaries; if False, return arrays
        
    Returns
    -------
    If return_dict is False:
        rate_templates : ndarray, shape (n_conditions, n_neurons, n_timepoints)
            Trial-averaged, smoothed firing rate templates
        behavior_templates : ndarray, shape (n_conditions, n_kin_vars, n_timepoints)
            Trial-averaged kinematic templates
        condition_list : ndarray
            Array of unique condition IDs
    If return_dict is True:
        Returns dictionaries as in standard_preprocessing()
    """
    
    # Call the standard preprocessing to get dictionaries
    rate_dict, behavior_dict, cond_list = standard_preprocessing(
        spikes, behavior, cond_ids,
        trial_alignment, trajectories_alignment,
        gaussian_sigma, bin_size,
        soft_norm, sampling_period,
        trial_dims, neural_dims, condition_dims
    )
    
    if return_dict:
        return rate_dict, behavior_dict, cond_list
    
    # Convert dictionaries to arrays for online MINT
    # Stack templates in order of condition IDs
    rate_templates = np.stack([rate_dict[c] for c in cond_list], axis=0)
    behavior_templates = np.stack([behavior_dict[c] for c in cond_list], axis=0)
    
    # Transpose to match expected format: (n_conditions, n_neurons/n_kin, n_timepoints)
    # The dictionaries already have the correct shape, so no transpose needed
    
    return rate_templates, behavior_templates, cond_list


def preprocessing_per_trial(
    spikes: npt.NDArray,
    behavior: npt.NDArray,
    cond_ids: npt.NDArray,
    trial_alignment: npt.NDArray,
    trajectories_alignment: npt.NDArray,
    gaussian_sigma: int,
    bin_size: int,
    soft_norm: float,
    sampling_period: float,
    trial_dims: int,
    neural_dims: int,
    condition_dims: int,
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    MINT preprocessing that returns per-trial arrays for online decoder.
    
    Performs PCA smoothing WITHOUT trial averaging, leaving that step
    for the fit() method to handle internally.
    
    Parameters
    ----------
    spikes : ndarray, shape (n_trials, n_neurons, n_timepoints)
        Raw spike count data
    behavior : ndarray, shape (n_trials, n_kin_vars, n_timepoints)
        Raw kinematic data  
    cond_ids : ndarray, shape (n_trials,)
        Condition ID for each trial
    trial_alignment : ndarray
        Time points (ms) relative to movement onset for trial data
    trajectories_alignment : ndarray
        Time points (ms) relative to movement onset for trajectory templates
    gaussian_sigma : int
        Gaussian filter sigma (ms)
    bin_size : int
        Bin size for data (ms)
    soft_norm : float
        PCA soft normalization constant
    sampling_period : float
        Sampling period (seconds)
    trial_dims : int
        Number of PCA components for trial-level reduction
    neural_dims : int
        Number of PCA components for neural reduction (or np.nan to skip)
    condition_dims : int
        Number of PCA components for condition-level reduction (or np.nan to skip)
        
    Returns
    -------
    rate_data : ndarray, shape (n_trials, n_neurons, n_timepoints)
        Per-trial smoothed firing rate data
    behavior_data : ndarray, shape (n_trials, n_kin_vars, n_timepoints)
        Per-trial kinematic data
    condition_ids : ndarray, shape (n_trials,)
        Condition ID for each trial
    """
    
    # Step 1: Smooth spikes with a Gaussian filter
    spikes_smooth = np.array(
        [utils.gauss_filt(spike, gaussian_sigma, bin_size) for spike in spikes]
    )
    
    # Step 2: Process kinematics (baseline alignment)
    behavior_processed = utils.process_kinematics(behavior, trial_alignment)
    
    # Step 3: Trim smoothed spikes and kinematics in time to match trajectory alignment
    t_mask = np.isin(trial_alignment, trajectories_alignment)
    spikes_smooth = spikes_smooth[:, :, t_mask]
    behavior_processed = behavior_processed[:, :, t_mask]
    
    # Step 4: Group by conditions for PCA smoothing
    cond_list = np.unique(cond_ids)
    X_dict = {cond: spikes_smooth[cond_ids == cond] for cond in cond_list}
    
    # Step 5: Apply PCA smoothing WITHOUT averaging
    X_dict_smoothed = utils.smooth_no_average(
        X_dict,
        soft_norm,
        bin_size,
        sampling_period,
        trial_dims,
        neural_dims,
        condition_dims  # Will apply condition PCA projection learned from averages
    )
    
    # Step 6: Convert back to array format
    # Reconstruct in original trial order
    rate_data = np.zeros_like(spikes_smooth)
    for cond in cond_list:
        cond_mask = cond_ids == cond
        rate_data[cond_mask] = X_dict_smoothed[cond]
    
    return rate_data, behavior_processed, cond_ids


def minimal_preprocessing(
    spikes: npt.NDArray,
    behavior: npt.NDArray,
    cond_ids: npt.NDArray,
    trial_alignment: npt.NDArray,
    trajectories_alignment: npt.NDArray,
) -> Tuple[Dict[int, npt.NDArray], Dict[int, npt.NDArray], npt.NDArray]:
    """
    Minimal preprocessing for pre-smoothed data (e.g., LFADS latents for mc_rtt).
    
    Performs only the essential steps:
    1. Process kinematics (baseline alignment) 
    2. Trim data to trajectory alignment window
    3. Group by conditions
    4. Compute trial averages
    
    Parameters
    ----------
    spikes : ndarray, shape (n_trials, n_neurons, n_timepoints)
        Pre-smoothed spike data (e.g., LFADS latents)
    behavior : ndarray, shape (n_trials, n_kin_vars, n_timepoints)
        Raw kinematic data
    cond_ids : ndarray, shape (n_trials,)
        Condition ID for each trial
    trial_alignment : ndarray
        Time points (ms) relative to movement onset for trial data
    trajectories_alignment : ndarray
        Time points (ms) relative to movement onset for trajectory templates
        
    Returns
    -------
    rate_templates : Dict[int, ndarray]
        Trial-averaged spike data by condition
    behavior_templates : Dict[int, ndarray] 
        Trial-averaged kinematic templates by condition
    condition_list : ndarray
        Array mapping condition indices to actual condition IDs
    """
    
    # Step 1: Process kinematics (baseline alignment)
    behavior_processed = utils.process_kinematics(behavior, trial_alignment)
    
    # Step 2: Trim data to match trajectory alignment
    t_mask = np.isin(trial_alignment, trajectories_alignment)
    spikes_trimmed = spikes[:, :, t_mask]
    behavior_processed = behavior_processed[:, :, t_mask]
    
    # Step 3: Group by conditions
    cond_list = np.unique(cond_ids)
    X = {cond: spikes_trimmed[cond_ids == cond] for cond in cond_list}
    Z_sort = {cond: behavior_processed[cond_ids == cond] for cond in cond_list}
    
    # Step 4: Compute trial averages
    rate_templates = {k: np.mean(v, axis=0) for k, v in X.items()}
    behavior_templates = {k: np.mean(v, axis=0) for k, v in Z_sort.items()}
    
    return rate_templates, behavior_templates, cond_list