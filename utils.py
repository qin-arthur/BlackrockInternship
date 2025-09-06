import numpy as np
import numpy.typing as npt
import numba as nb
import warnings

from typing import Tuple, Dict, Literal, List, Callable, Sequence
from scipy import signal
from scipy.stats import poisson
from sklearn.decomposition import PCA

def process_kinematics(
    behavior: npt.NDArray,
    trial_alignment: npt.NDArray
) -> npt.NDArray:
    """
    Aligns kinematic data to movement onset and returns updated data plus labels.

    Parameters
    ----------
    behavior : ndarray, shape (n_trials, n_kin_vars, T)
        Raw kinematic variables.
    trial_alignment : ndarray, shape (T,)
        Timepoints (ms) relative to movement onset, must contain a 0 entry.

    Returns
    -------
    aligned_behavior : ndarray, same shape as `behavior`
        Behavior shifted so that positions at time 0 become the new origin.
    labels : List[str]
        Names of the kinematic channels: ["xpos", "ypos", "xvel", "yvel"].
    """
    behavior = behavior.copy()
    # Calculate position relative to onset
    onset_idx = np.where(trial_alignment == 0)[0][0]
    behavior[:, :2] -= behavior[:, :2, onset_idx:onset_idx+1]


    return behavior


def smooth_average(
    spikes: Dict[int, npt.NDArray],
    soft_norm: float,
    bin_size: int,
    sampling_period: float,
    trial_dims: int,
    neural_dims: int,
    condition_dims: int,
) -> Dict[int, npt.NDArray]:
    # Compute mean and max firing rates from an initial averaging of rates across trials.
    s_avg = {k: np.mean(v, axis=0) for k, v in spikes.items()}
    
    # Calculate the average firing rate for each neuron across all condition-averaged trajectories.
    s_avg_wide = np.hstack([v for v in s_avg.values()])
    mu = np.mean(s_avg_wide, axis=1, keepdims=True)
    soft_norm = soft_norm * bin_size * sampling_period
    norm_factor = 1 / (soft_norm + np.max(s_avg_wide, axis=1, keepdims=True))
    
    # Mean-center and soft-normalize single trial rates
    spikes = {k: (v - mu[None, ...]) * norm_factor[None, ...] for k, v in spikes.items()}
    
    # Use dimensionality reduction to smooth across trials within each condition.
    def _trialwise_pca(x: np.ndarray):
        in_shape = x.shape
        x = x.reshape((x.shape[0], -1)).T  # Reshape to (trials, neurons * time) then transpose
        pca = PCA(n_components=trial_dims)
        pca.fit(x)
        M = pca.components_
        x = np.dot((M * M.T), x.T)
        return x.reshape(in_shape)
    
    spikes = {c: _trialwise_pca(x) for c, x in spikes.items()}
    
    # Average rates across trials within each condition.
    spikes_bar = {c: np.mean(x, axis=0) for c, x in spikes.items()}
    
    
    # Use dimensionality reduction to smooth across neurons.
    if ~np.isnan(neural_dims):
        X_neuron_bar = np.vstack([v.T for v in spikes_bar.values()])
        # Reduce neurons dimensionality with PCA
        pca = PCA(n_components=min(neural_dims, X_neuron_bar.shape[1]))
        pca.fit(X_neuron_bar)
        proj_mat = np.dot(pca.components_.T, pca.components_)
        spikes_bar = {c: np.dot(proj_mat, x) for c, x in spikes_bar.items()}

    # Use dimensionality reduction to smooth across conditions.
    if ~np.isnan(condition_dims):
        spikes_bar_nt = np.stack([x for x in spikes_bar.values()], axis=0)
        n_conds, n_neurons, n_times = spikes_bar_nt.shape
        spikes_bar_nt = spikes_bar_nt.reshape((n_conds, n_neurons * n_times))
        pca = PCA(n_components=min(condition_dims, n_conds))
        pca.fit(spikes_bar_nt.T)
        M = pca.components_
        proj_mat = np.matmul(M.T, M)
        spikes_bar_nt = np.matmul(proj_mat, spikes_bar_nt)
        spikes_bar_array = np.reshape(spikes_bar_nt, (n_conds, n_neurons, n_times))
        
        # Undo mean-centering and soft-normalization for array
        spikes_bar_array /= norm_factor[None, ...]
        spikes_bar_array += mu[None, ...]
        np.clip(spikes_bar_array, a_min=0, a_max=np.inf, out=spikes_bar_array)
        
        # Convert back to dictionary
        spikes_bar = {k: spikes_bar_array[i] for i, k in enumerate(spikes.keys())}
    else:
        # Undo mean-centering and soft-normalization for dictionary
        for k, v in spikes_bar.items():
            v /= norm_factor
            v += mu
            np.clip(v, a_min=0, a_max=np.inf, out=v)
            spikes_bar[k] = v

    return spikes_bar


def smooth_no_average(
    spikes: Dict[int, npt.NDArray],
    soft_norm: float,
    bin_size: int,
    sampling_period: float,
    trial_dims: int,
    neural_dims: int,
    condition_dims: int,
) -> Dict[int, npt.NDArray]:
    """Apply PCA smoothing without trial averaging.
    
    Preserves trial-to-trial variability for online decoding.
    
    Args:
        spikes: Spike counts by condition (n_trials, n_neurons, n_time)
        soft_norm: Normalization constant
        bin_size: Bin size in ms
        sampling_period: Sampling period in seconds
        trial_dims: Trial PCA dimensions
        neural_dims: Neural PCA dimensions (np.nan to skip)
        condition_dims: Condition PCA dimensions (np.nan to skip)
        
    Returns:
        Smoothed per-trial spike data
    """
    # Compute mean and max firing rates from an initial averaging of rates across trials.
    s_avg = {k: np.mean(v, axis=0) for k, v in spikes.items()}
    
    # Calculate the average firing rate for each neuron across all condition-averaged trajectories.
    s_avg_wide = np.hstack([v for v in s_avg.values()])
    mu = np.mean(s_avg_wide, axis=1, keepdims=True)
    soft_norm = soft_norm * bin_size * sampling_period
    norm_factor = 1 / (soft_norm + np.max(s_avg_wide, axis=1, keepdims=True))
    
    # Mean-center and soft-normalize single trial rates
    spikes = {k: (v - mu[None, ...]) * norm_factor[None, ...] for k, v in spikes.items()}
    
    # Use dimensionality reduction to smooth across trials within each condition.
    def _trialwise_pca(x: np.ndarray):
        in_shape = x.shape
        x = x.reshape((x.shape[0], -1)).T  # Reshape to (trials, neurons * time) then transpose
        pca = PCA(n_components=trial_dims)
        pca.fit(x)
        M = pca.components_
        x = np.dot((M * M.T), x.T)
        return x.reshape(in_shape)
    
    spikes = {c: _trialwise_pca(x) for c, x in spikes.items()}
    
    # Skip trial averaging for online decoder
    
    # Apply neural-level PCA smoothing per trial
    if ~np.isnan(neural_dims):
        # Create projection matrix using trial-averaged data for fitting
        spikes_bar = {c: np.mean(x, axis=0) for c, x in spikes.items()}
        X_neuron_bar = np.vstack([v.T for v in spikes_bar.values()])
        # Reduce neurons dimensionality with PCA
        pca = PCA(n_components=min(neural_dims, X_neuron_bar.shape[1]))
        pca.fit(X_neuron_bar)
        proj_mat = np.dot(pca.components_.T, pca.components_)
        # Apply projection to each trial
        spikes = {c: np.array([np.dot(proj_mat, trial) for trial in x]) for c, x in spikes.items()}

    # Apply condition-level PCA smoothing
    if ~np.isnan(condition_dims):
        # First, compute trial averages to learn the PCA projection
        spikes_bar = {c: np.mean(x, axis=0) for c, x in spikes.items()}
        spikes_bar_nt = np.stack([x for x in spikes_bar.values()], axis=0)
        n_conds, n_neurons, n_times = spikes_bar_nt.shape
        spikes_bar_nt_reshaped = spikes_bar_nt.reshape((n_conds, n_neurons * n_times))
        
        # Fit PCA on condition-averaged data
        pca = PCA(n_components=min(condition_dims, n_conds))
        pca.fit(spikes_bar_nt_reshaped.T)
        M = pca.components_
        proj_mat = np.matmul(M.T, M)
        
        # Apply the projection to the averaged data
        spikes_bar_proj = np.matmul(proj_mat, spikes_bar_nt_reshaped)
        spikes_bar_proj = spikes_bar_proj.reshape((n_conds, n_neurons, n_times))
        
        # For each trial, apply the same transformation that was applied to its condition average
        # This preserves the condition-level smoothing while keeping per-trial variations
        spikes_smoothed = {}
        for cond_idx, (k, v) in enumerate(spikes.items()):
            # Compute smoothing difference from condition average
            avg_diff = spikes_bar_proj[cond_idx] - spikes_bar[k]
            
            # Apply to each trial while preserving variability
            trials_smoothed = []
            for trial in v:
                trial_smoothed = trial + avg_diff
                trials_smoothed.append(trial_smoothed)
            spikes_smoothed[k] = np.array(trials_smoothed)
        
        spikes = spikes_smoothed
    
    # Restore original scale for each trial
    for k, v in spikes.items():
        v /= norm_factor[None, ...]
        v += mu[None, ...]
        np.clip(v, a_min=0, a_max=np.inf, out=v)
        spikes[k] = v

    return spikes


def gauss_filt(
    spikes: npt.NDArray, 
    gaussian_sigma: int, 
    bin_size: float
) -> npt.NDArray:
    """Apply Gaussian smoothing to spike data.
    
    Args:
        spikes: Spike counts (n_neurons, n_time)
        gaussian_sigma: Kernel width in ms
        bin_size: Bin size
        
    Returns:
        Smoothed spike data
    """

    # Create Gaussian kernel
    width = 4
    N = 2 * width * gaussian_sigma + 1
    alpha = (N - 1) / (2 * gaussian_sigma)
    y = signal.windows.gaussian(M=N, std=gaussian_sigma)
    y = y / sum(y) * bin_size

    # Pad signal edges with mean values
    pre = np.matlib.repmat(spikes[:, 0:gaussian_sigma].mean(axis=1), width * gaussian_sigma, 1).T
    post = np.matlib.repmat(spikes[:, -gaussian_sigma:].mean(axis=1), width * gaussian_sigma, 1).T
    inputVal = np.concatenate([pre, np.double(spikes), post], axis=1)

    # Apply convolution
    filt_spikes = np.zeros(inputVal.shape)
    for i in range(len(inputVal)):
        c = np.convolve(inputVal[i, :], y)
        filt_spikes[i, :] = c[width * gaussian_sigma : -width * gaussian_sigma]

    # Remove padding
    filt_spikes = filt_spikes[:, width * gaussian_sigma : -width * gaussian_sigma]

    return filt_spikes



def bin_data(
    data: npt.NDArray,
    bin_size: int,
    method: Literal["mean", "sum"],
) -> npt.NDArray:
    """Bin data using specified aggregation method.
    
    Args:
        data: Input data (n_features, n_time)
        bin_size: Number of samples per bin
        method: 'mean' or 'sum' aggregation
        
    Returns:
        Binned data (n_features, n_bins)
    """
    n_bins = int(np.floor(data.shape[1] / bin_size))
    binned_data = np.empty((data.shape[0], n_bins))
    if method == "mean":
        for i in range(n_bins):
            binned_data[:, i] = np.mean(data[:, bin_size * i : bin_size * (i + 1)], 1)
    elif method == "sum":
        for i in range(n_bins):
            binned_data[:, i] = np.sum(data[:, bin_size * i : bin_size * (i + 1)], 1)
    else:
        np.error_message("Unrecogonised binning method")

    return binned_data


def get_rate_indices(lamda, lambda_range, nRates):
    """Quantize continuous rates to discrete indices.
    
    Args:
        lamda: Continuous firing rates
        lambda_range: (min, max) rate bounds
        nRates: Number of quantization bins
        
    Returns:
        Quantized rate indices
    """
    lambda_min = lambda_range[0]
    lambdaMax = lambda_range[1]
    lamda[lamda < lambda_min] = lambda_min
    lamda[lamda > lambdaMax] = lambdaMax

    # Convert from rates to indices.
    v = (lamda - lambda_min) / np.diff(lambda_range) * (nRates - 1) + 1
    # Round and convert to uint16.
    v = np.rint(v).astype(int)

    return v


def build_poisson(
    rate_range: Tuple[float, float],
    num_rate_bins: int,
    bin_duration: float,
    min_prob: float
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Build Poisson log-likelihood lookup table for efficient inference.
    
    Args:
        rate_range: (min, max) firing rates
        num_rate_bins: Number of discrete rate bins
        bin_duration: Duration of each bin
        min_prob: Minimum probability floor
        
    Returns:
        log_likelihood_lookup: Log-likelihood table (num_rate_bins, max_spikes+1)
        rate_bin_centers: Centers of rate bins
    """
    # Build rate bin centers
    rates = np.linspace(rate_range[0], rate_range[1], num_rate_bins)
    rates = np.array(rates).reshape(num_rate_bins, 1)

    # Determine maximum spike count
    max_spikes = round(bin_duration * 1000)

    # Build count and rate matrices
    counts_mat = np.matlib.repmat(np.arange(0, max_spikes + 1), num_rate_bins, 1)
    rates_mat  = np.matlib.repmat(rates, 1, max_spikes + 1)

    # Compute normalized Poisson probabilities
    L = poisson.pmf(counts_mat, rates_mat)
    L[L <= min_prob] = np.nan
    norm = 1 - min_prob * np.sum(np.isnan(L), axis=1)
    L = L * np.array(norm / np.nansum(L, axis=1)).reshape(num_rate_bins, 1)
    L[np.isnan(L)] = min_prob
    return np.log(L), rates

def recursion(
    Q, s_new, s_old, t_prime, L, V, first_idx, shifted_idx1, shifted_idx2, N, tau_prime
):
    """Non-JIT version of recursive Bayesian update.
    
    Args:
        Q: Log-posterior
        s_new: Current spike counts
        s_old: Lagged spike counts
        t_prime: Current time index
        L: Log-likelihood table
        V: Rate indices
        first_idx: Condition start indices
        shifted_idx1: Past state indices
        shifted_idx2: Future state indices
        N: Number of neurons
        tau_prime: History window size
        
    Returns:
        Updated log-posterior
    """
    # Time update
    Q = np.append(0, Q[0:-1])
    Q[first_idx] = 0

    # Measurement update
    if t_prime > tau_prime + 1:
        for n in range(N):
            Q = Q + L[V[:, n] - 1, int(s_new[n])]
            Q[shifted_idx2] = Q[shifted_idx2] - L[V[shifted_idx1, n] - 1, int(s_old[n])]
    else:
        for n in range(N):
            Q = Q + L[V[:, n] - 1, int(s_new[n])]

    return Q


@nb.njit(fastmath=True, parallel=True, cache=True)
def recursion_jit(
    Q: npt.NDArray, 
    s_new: npt.NDArray, 
    s_old: npt.NDArray, 
    t_prime: int, 
    L: npt.NDArray, 
    V: npt.NDArray, 
    first_idx: npt.NDArray, 
    shifted_idx1: npt.NDArray, 
    shifted_idx2: npt.NDArray, 
    N: int, 
    tau_prime: int
) -> npt.NDArray:
    """JIT-compiled recursive Bayesian update for performance.
    
    Optimized version using Numba for real-time decoding.
    """
    # Shift posterior forward in time
    Q = np.append(0, Q[0:-1])
    # Reset at condition boundaries
    Q[first_idx] = 0

    # Measurement update
    if t_prime > tau_prime + 1:
        # Update with new observations
        for n in nb.prange(N):
            Q += L[V[:, n] - 1, int(s_new[n])]
        # Correct for old observations leaving window
        for n in range(N):
            Q[shifted_idx2] -= L[V[shifted_idx1, n] - 1, int(s_old[n])]
    else:
        # Only add new observations (no history to remove)
        for n in nb.prange(N):
            Q += L[V[:, n] - 1, int(s_new[n])]
    return Q

@nb.njit(fastmath=True, parallel=True, cache=True)
def recursion_broken(
    Q: npt.NDArray, 
    s_new: npt.NDArray, 
    s_old: npt.NDArray, 
    t_prime: int, 
    L: npt.NDArray, 
    V: npt.NDArray, 
    first_idx: npt.NDArray, 
    shifted_idx1: npt.NDArray, 
    shifted_idx2: npt.NDArray, 
    N: int, 
    tau_prime: int
) -> npt.NDArray:
    """Alternative JIT recursion implementation (experimental)."""
    Q = np.append(0, Q[0:-1])
    Q[first_idx] = 0

    if t_prime > tau_prime + 1:
        for n in nb.prange(N):
            Q += L[V[:, n] - 1, int(s_new[n])]
            Q[shifted_idx2[n]] -= L[V[shifted_idx1[n], n] - 1, int(s_old[n])]
    else:
        for n in nb.prange(N):
            Q += L[V[:, n] - 1, int(s_new[n])]
    return Q

def get_time_indices_online(
    bin_idx: int,
    history_bins: int,
):
    """Get time indices for online causal decoding.
    
    Args:
        bin_idx: Current bin index
        history_bins: Size of history window
        
    Returns:
        time_indices: Array of time indices
        state_index_fn: Function mapping state to time indices
    """
    start = max(0, bin_idx - history_bins)
    time_indices = np.arange(start, bin_idx + 1)

    def state_index_fn(k_prime: np.ndarray) -> np.ndarray:
        # Always align to *current* bin, not nominal history length.
        # Works both before and after warm-up.
        return (k_prime - bin_idx) + time_indices

    return state_index_fn


def get_time_indices(
    bin_idx: int,
    num_bins: int,
    trial_length: int,
    bin_size: int,
    history_bins: int,
    causal: bool,
) -> Tuple[npt.NDArray, Callable[[npt.NDArray], npt.NDArray]]:
    """Map bin index to full-rate time points for offline decoding.
    
    Args:
        bin_idx: Current bin index
        num_bins: Total bins per trial
        trial_length: Total timepoints
        bin_size: Samples per bin
        history_bins: History window size
        causal: Use trailing vs centered window
        
    Returns:
        time_indices: Full-rate timepoints
        state_index_fn: State-to-time mapping function
    """
    # Compute window center or trailing point
    t_center = (bin_idx + 1) * bin_size
    if not causal:
        # center window around t_center
        half_window = (history_bins + 1) * bin_size - 1
        adjustment = round((half_window + 1 + bin_size) / 2)
        t_center -= adjustment

    # Build full-rate indices
    time_indices = t_center + np.arange(0, bin_size)

    # at first valid bin, prepend early times so estimates flow backwards
    if bin_idx == history_bins + 1:
        prepend_count = time_indices[0] - 1
        time_indices = np.append(
            np.arange(1, prepend_count + 1), time_indices
        )

    # at last bin, append trailing times to reach trial end
    if (bin_idx + 1) == num_bins and time_indices[-1] < trial_length:
        append_count = trial_length - time_indices[-1]
        time_indices = np.append(
            time_indices,
            time_indices[-1] + np.arange(1, append_count)
        )

    # truncate any overshoot
    time_indices = time_indices[time_indices <= trial_length]


    # Create state-to-time mapping
    def state_index_fn(k_prime: npt.NDArray) -> npt.NDArray:
        """
        Convert a raw trajectory index k′ into full‐rate timepoints
        aligned with this bin's estimate.
        """
        return ((k_prime - bin_idx) * bin_size + time_indices) - 1

    return time_indices - 1, state_index_fn


def flat_to_cond_state(flat_index, start_indices):
    """Convert flat state index to (condition, local_state).
    
    Args:
        flat_index: Global state index
        start_indices: Start indices per condition
        
    Returns:
        (condition, local_state) tuple
    """
    cond = np.where(flat_index >= start_indices)[0][-1]
    local = flat_index - start_indices[cond]
    return cond, local


def cond_state_to_flat(cond, local_state, start_indices):
    """Convert (condition, local_state) to flat index.
    
    Args:
        cond: Condition index
        local_state: State within condition
        start_indices: Start indices per condition
        
    Returns:
        Flat state index
    """
    return start_indices[cond] + local_state

def get_simple_state_indices(k_prime_hats, trajectory_length):
    """Map state indices for online decoding.
    
    Args:
        k_prime_hats: Adjacent state indices [k1, k2]
        trajectory_length: Maximum valid index
        
    Returns:
        Clamped state indices (2, 1)
    """
    k_idx = np.array([[max(0, min(k_prime_hats[0], trajectory_length-1))],
                      [max(0, min(k_prime_hats[1], trajectory_length-1))]])
    return k_idx

def maximum_likelihood(Q, tau_prime, first_idx, first_tau_prime_idx, restrictedConds):
    """Find maximum likelihood state from posterior.
    
    Args:
        Q: Log-posterior
        tau_prime: History window size
        first_idx: Condition start indices
        first_tau_prime_idx: History indices
        restrictedConds: Conditions to exclude
        
    Returns:
        c_hat: Most likely condition
        k_prime_hats: Adjacent state indices
    """
    # Get trajectory lengths
    K = np.append(first_idx[1:], len(Q)) - first_idx - 1  # -1 for Python indexing

    # Apply state restrictions
    Q[first_tau_prime_idx] = np.nan
    if not len(restrictedConds) == 0:
        for i in range(len(restrictedConds)):
            c = restrictedConds[i]
            Q[cond_state_to_flat(c, np.arange(0, K[c] + 1), first_idx)] = (
                np.nan
            )  # 0 for python indexing, +1 for the open brackets

    # Find maximum likelihood state
    idx = np.where(Q == np.nanmax(Q))[0][0]
    c_hat, k1 = flat_to_cond_state(idx, first_idx)

    # Select adjacent state for interpolation
    Q_c = Q[cond_state_to_flat(c_hat, np.arange(K[c_hat] + 1), first_idx)]  # +1 for open brackets
    if tau_prime + 1 < k1 < K[c_hat]:
        if Q_c[k1 - 1] > Q_c[k1 + 1]:
            k2 = k1 - 1
        else:
            k2 = k1 + 1
    elif k1 > tau_prime + 1:
        k2 = k1 - 1
    else:
        k2 = k1 + 1

    k_prime_hats = np.append(k1, k2)

    return c_hat, k_prime_hats


def fit_poisson_interp(spike_counts, rate1, rate2, max_iters, tolerance, default_alpha):
    """Fit interpolation parameter between two rate templates.
    
    Uses Newton's method to find optimal mixing weight.
    
    Args:
        spike_counts: Observed spikes
        rate1: First rate template
        rate2: Second rate template
        max_iters: Maximum iterations
        tolerance: Convergence threshold
        default_alpha: Default mixing weight
        
    Returns:
        Optimal interpolation weight alpha
    """
    # Ensure shape compatibility
    if spike_counts.shape != rate1.shape:
        # If shapes are transposed, transpose spike_counts
        if spike_counts.shape == rate1.T.shape:
            spike_counts = spike_counts.T
        else:
            raise ValueError(f"Shape mismatch: spike_counts {spike_counts.shape}, rates {rate1.shape}")
    
    delta_rates = rate2 - rate1
    total_delta = np.sum(delta_rates)

    alpha = 0.5
    i = 0
    while i < max_iters:
        # Compute gradient and Hessian
        frac = delta_rates / (rate1 + alpha * delta_rates)
        gradient = np.sum(spike_counts * frac) - total_delta
        hessian  = -np.sum(spike_counts * (frac ** 2))

        # Newton–Raphson update
        step = gradient / hessian
        alpha -= step

        # Check convergence or bounds
        if abs(step) < tolerance or alpha < 0 or alpha > 1:
            alpha = max(min(alpha, 1), 0)
            return alpha
        i += 1

    # Handle non-convergence
    if np.isnan(alpha):
        warnings.warn("Poisson interpolation failed; using default alpha.")
        return default_alpha

    return alpha


def get_state_indices(k_prime_hats, f, K):
    """Convert state indices using mapping function.
    
    Args:
        k_prime_hats: Adjacent state indices
        f: State-to-time mapping function
        K: Maximum valid index
        
    Returns:
        Clamped state indices
    """
    k_idx = np.array((f(k_prime_hats[0]), f(k_prime_hats[1])))
    
    # Clamp to valid range
    k_idx[k_idx > K] = K
    k_idx[k_idx < 0] = 0

    return k_idx