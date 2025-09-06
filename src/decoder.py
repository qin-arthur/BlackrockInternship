import numpy as np
import numpy.typing as npt
from . import utils

import json
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Tuple, List, Dict, Callable, Union

@dataclass
class MINTSettings:
    """Configuration for offline MINT decoder (batch processing).
    
    Handles both raw data (with preprocessing) and preprocessed template workflows.
    For preprocessed data, only core decoding parameters are needed.
    For raw data, additional preprocessing parameters are required.
    """
    # Required fields (no defaults)
    task: str                             # Task identifier
    data_path: str                        # Input data directory
    results_path: str                     # Output directory
    bin_size: int                         # Bin size in ms
    observation_window: int               # Decoding window in ms
    causal: bool                          # True=past-only, False=centered window
    test_alignment: npt.NDArray           # Time points for evaluation (ms)
    min_lambda: float                     # Minimum firing rate threshold
    
    # Optional preprocessing parameters (for raw data workflows)
    trial_alignment: npt.NDArray | None = None          # Trial alignment points (ms)
    trajectories_alignment: npt.NDArray | None = None   # Template alignment points (ms)
    gaussian_sigma: int | None = None                   # Smoothing kernel width (ms)
    neural_dims: int | None = None                      # Neural PCA dimensions
    condition_dims: int | None = None                   # Condition PCA dimensions
    trial_dims: int | None = None                       # Trial PCA dimensions
    
    # Optional fields with defaults
    sampling_period: float = 0.001        # Sampling period (seconds)
    soft_norm: float = 5.0                # PCA normalization constant
    min_prob: float = 1e-6                # Minimum probability threshold
    min_rate: float = 0.0                 # Minimum firing rate (spikes/sec)
    interp_mode: int = 2                  # Interpolation mode (0=none, 2=condition)
    interp_max_iters: int = 10            # Max interpolation iterations
    interp_tolerance: float = 0.01        # Interpolation convergence threshold
    num_rate_bins: int = 2000             # Number of rate quantization bins
    
    # Computed fields (set during __init__)
    rate_range_lookup: Tuple[float, float] = field(init=False)  # Rate range (spikes/sec)
    min_rate_per_bin: float = field(init=False)            # Min rate per bin
    history_bins: int = field(init=False)                  # History window size
    log_likelihood_lookup: npt.NDArray = field(init=False) # Poisson lookup table
    rate_bin_centers: npt.NDArray = field(init=False)      # Rate bin centers


@dataclass
class MINTState:
    """Decoder state after training, containing templates and indices.
    
    Stores trial-averaged templates and precomputed indices for efficient
    recursive Bayesian inference during decoding.
    """
    
    rate_templates: Dict[int, npt.NDArray]       # Neural templates by condition
    behavior_templates: Dict[int, npt.NDArray]   # Kinematic templates by condition
    rate_indices: npt.NDArray                    # Quantized rate indices (V matrix)
    base_state_indices: npt.NDArray              # Start indices per condition
    lagged_state_indices: npt.NDArray            # History window indices
    shifted_indices_past: npt.NDArray            # Past state lookups
    shifted_indices_future: npt.NDArray          # Future state lookups
    interp_map: npt.NDArray | None = None        # Interpolation weights
    condition_list: npt.NDArray | None = None    # Condition ID mapping
    kinematic_labels: Tuple[str] = ("xpos", "ypos", "xvel", "yvel")


class MINTDecoder:
    """Offline MINT decoder for batch neural decoding.
    
    Processes all trials simultaneously (batch mode) with save/load support.
    Can work with either raw data (applies preprocessing) or preprocessed templates.
    
    Example:
        # With raw data
        decoder = MINTDecoder(settings)
        decoder.fit(spikes, behavior, conditions)
        
        # With preprocessed templates
        decoder = MINTDecoder(settings)
        decoder.fit(rate_templates=templates, ...)
    """

    def __init__(self, settings: MINTSettings | None = None, state: MINTState | None = None, load_from_path: str | None = None) -> None:
        """Initialize decoder with settings, state, or load from disk.
        
        Args:
            settings: Configuration parameters
            state: Pre-trained decoder state
            load_from_path: Path to saved decoder
        """
        if load_from_path is not None:
            self._load_from_disk(load_from_path)
            return
            
        if settings is None:
            raise ValueError("Either 'settings' or 'load_from_path' must be provided")
            
        # Initialize settings and state
        self.settings = settings
        self.state = state if state is not None else MINTState(
            rate_templates={},
            behavior_templates={},
            rate_indices=None,
            base_state_indices=None,
            lagged_state_indices=None,
            shifted_indices_past=None,
            shifted_indices_future=None
        )
        
        # Compute derived settings
        bin_duration = self.settings.bin_size * self.settings.sampling_period
        self.settings.history_bins = (
            round(self.settings.observation_window / self.settings.bin_size) - 2
        )
        
        self.settings.rate_range_lookup = [x * bin_duration for x in [self.settings.min_lambda, 500]]
        
        # Build Poisson log-likelihood lookup table
        self.settings.log_likelihood_lookup, self.settings.rate_bin_centers = utils.build_poisson(
            self.settings.rate_range_lookup,
            self.settings.num_rate_bins,
            bin_duration,
            self.settings.min_prob,
        )
        
        # Convert minimum rate from spikes/sec to spikes/bin
        self.settings.min_rate_per_bin = self.settings.min_rate * bin_duration
        

    def fit(
        self,
        spikes: npt.NDArray = None,
        behavior: npt.NDArray = None,
        cond_ids: npt.NDArray = None,
        rate_templates: Dict[int, npt.NDArray] = None,
        behavior_templates: Dict[int, npt.NDArray] = None,
        condition_list: npt.NDArray = None,
    ) -> "MINTDecoder":
        """Train decoder on neural and behavioral data.
        
        Supports two input modes:
        1. Raw data: Applies preprocessing pipeline
        2. Templates: Uses pre-computed templates directly
        
        Args:
            spikes: Raw spikes (n_trials, n_neurons, n_time)
            behavior: Raw kinematics (n_trials, n_channels, n_time)
            cond_ids: Trial conditions (n_trials,)
            rate_templates: Pre-computed neural templates
            behavior_templates: Pre-computed kinematic templates
            condition_list: Condition ID mapping
            
        Returns:
            Self for method chaining
        """
        # Determine input mode and get templates
        if rate_templates is not None and behavior_templates is not None and condition_list is not None:
            # Template mode: Use provided preprocessed templates
            self.state.rate_templates = rate_templates
            self.state.behavior_templates = behavior_templates  
            self.state.condition_list = condition_list
            
        elif spikes is not None and behavior is not None and cond_ids is not None:
            # Raw data mode: Apply standard preprocessing
            from . import preprocessing
            
            # Validate required preprocessing parameters
            required_params = [
                'trial_alignment', 'trajectories_alignment', 'gaussian_sigma', 
                'neural_dims', 'condition_dims', 'trial_dims'
            ]
            missing_params = [p for p in required_params if getattr(self.settings, p) is None]
            if missing_params:
                raise ValueError(
                    f"For raw data preprocessing, the following MINTSettings parameters are required: {missing_params}\n"
                    "Either provide these parameters, or use preprocessed templates instead."
                )
            
            self.state.rate_templates, self.state.behavior_templates, self.state.condition_list = (
                preprocessing.standard_preprocessing(
                    spikes=spikes,
                    behavior=behavior,
                    cond_ids=cond_ids,
                    trial_alignment=self.settings.trial_alignment,
                    trajectories_alignment=self.settings.trajectories_alignment,
                    gaussian_sigma=self.settings.gaussian_sigma,
                    bin_size=self.settings.bin_size,
                    soft_norm=self.settings.soft_norm,
                    sampling_period=self.settings.sampling_period,
                    trial_dims=self.settings.trial_dims,
                    neural_dims=self.settings.neural_dims,
                    condition_dims=self.settings.condition_dims,
                )
            )
        else:
            raise ValueError(
                "Must provide either:\n"
                "1. Raw data: spikes, behavior, cond_ids, OR\n" 
                "2. Preprocessed templates: rate_templates, behavior_templates, condition_list"
            )
        # Bin templates to match decoding resolution
        Lambda = {
            c: utils.bin_data(x, self.settings.bin_size, "mean") 
            for c, x in self.state.rate_templates.items()
        }
        
        # Quantize rates to discrete indices
        self.state.rate_indices = {
            c: utils.get_rate_indices(x, self.settings.rate_range_lookup, self.settings.num_rate_bins)
            for c, x in Lambda.items()
        }
        
        # Build cumulative indices for state space
        rate_indice_shapes = [v.shape[1] for v in self.state.rate_indices.values()]
        self.state.base_state_indices = np.cumsum([0] + rate_indice_shapes[:-1])
        state_end_indices = np.cumsum(rate_indice_shapes)
        
        # Create history window indices
        tmp = (np.arange(self.settings.history_bins)[:, None] + 
               self.state.base_state_indices[None, :]).flatten()
        self.state.lagged_state_indices = np.sort(tmp)
        
        # Flatten rate indices for efficient access
        n_neurons = self.state.rate_indices[list(self.state.rate_indices.keys())[0]].shape[0]
        self.state.rate_indices = np.stack(
            [v.T for v in self.state.rate_indices.values()], axis=0
        ).reshape((-1, n_neurons))
    
        # Create shifted indices for recursion
        self.state.shifted_indices_past = np.hstack([
            np.arange(
                self.state.base_state_indices[_],
                state_end_indices[_] - self.settings.history_bins
            )
            for _ in range(len(Lambda))
        ])
        self.state.shifted_indices_future = np.hstack([
            np.arange(
                self.state.base_state_indices[_] + self.settings.history_bins,
                state_end_indices[_]
            )
            for _ in range(len(Lambda))
        ])        
            
        
        return self
        
        
    def predict(self, spike_trials):
        """Decode neural and behavioral states from spike data.
        
        Processes all trials in batch using recursive Bayesian inference.
        
        Args:
            spike_trials: Spike counts (n_trials, n_neurons, n_time)
            
        Returns:
            decoded_rates: Estimated firing rates
            decoded_behavior: Estimated kinematics
        """
        # Extract trial dimensions
        T = np.array([_.shape[1] for _ in spike_trials])
        n_early_samples = (
            self.settings.bin_size * (self.settings.history_bins + 2) - 1
        )
        
        # Initialize output arrays
        n_trials = len(spike_trials)
        X_hat = np.zeros(spike_trials.shape)
        Z_hat = np.zeros((spike_trials.shape[0], 4, spike_trials.shape[2]))

        # Bin spikes
        S_bar = np.empty(
            (spike_trials.shape[0], spike_trials.shape[1], int(spike_trials.shape[2] / self.settings.bin_size))
        )
        for i in range(len(S_bar)):
            S_bar[i] = utils.bin_data(spike_trials[i], self.settings.bin_size, "sum")

        # Initialize C_hat, K_hat and Alpha_hat depending on interp_mode
        if self.settings.interp_mode == 2:
            C_hat = np.zeros((n_trials, 2, spike_trials.shape[2]))
            K_hat = np.zeros((n_trials, 4, spike_trials.shape[2]))
            Alpha_hat = np.zeros((n_trials, 3, spike_trials.shape[2]))
        else:
            C_hat = np.zeros((n_trials, 1, spike_trials.shape[2]))
            K_hat = np.zeros((n_trials, 1, spike_trials.shape[2]))
            Alpha_hat = np.zeros((n_trials, 1, spike_trials.shape[2]))

        # For each trial
        for tr in range(n_trials):
            # Reset Q and preallocate outputs
            Q = np.zeros((self.state.rate_indices.shape[0], 1))

            # For each time bin
            T_prime = S_bar.shape[2]
            for t_prime in range(T_prime):
                s_new = S_bar[tr, :, t_prime]
                if t_prime > (self.settings.history_bins + 1):
                    s_old = S_bar[tr, :, t_prime - self.settings.history_bins - 2]
                else:
                    s_old = np.zeros(s_new.shape)
                    
                first_template = next(iter(self.state.rate_templates.values()))
                N = first_template.shape[0]

                # Advanced log probabilities recursion
                Q = utils.recursion(
                    Q,
                    s_new,
                    s_old,
                    t_prime,
                    self.settings.log_likelihood_lookup,
                    self.state.rate_indices,
                    self.state.base_state_indices,
                    self.state.shifted_indices_past,
                    self.state.shifted_indices_future,
                    N,
                    self.settings.history_bins,
                )

                # Perform decoding when sufficient history exists
                if t_prime > self.settings.history_bins:
                    t_idx, f = utils.get_time_indices(
                        t_prime,
                        T_prime,
                        T[tr],
                        self.settings.bin_size,
                        self.settings.history_bins,
                        self.settings.causal,
                    )

                    S_curr = np.double(S_bar[tr, :, t_prime - self.settings.history_bins - 1 : t_prime + 1])
                    (
                        X_hat[tr, :, t_idx],
                        Z_hat[tr, :, t_idx],
                        C_hat[tr, :, t_idx],
                        K_hat[tr, :, t_idx],
                        Alpha_hat[tr, :, t_idx],
                    ) = self.estimate(Q.copy(), S_curr, f)

            # Mark acausal estimates as NaN if using causal mode
            if self.settings.causal:
                X_hat[tr, :, :n_early_samples] = np.nan
                Z_hat[tr, :, :n_early_samples] = np.nan

        # Apply minimum firing rate constraint
        np.clip(X_hat, min=self.settings.min_rate_per_bin, max=np.inf, out=X_hat)

        return X_hat, Z_hat
    
    
    def estimate(
        self,
        log_posterior,
        spike_counts,
        time_index_fn,
    ):
        """Estimate states from posterior distribution.
        
        Selects most likely states and applies interpolation.
        
        Args:
            log_posterior: Current log-posterior (n_states, 1)
            spike_counts: Observation window (n_neurons, window_bins)
            time_index_fn: Function to map state to time indices
            
        Returns:
            rate_estimate: Decoded rates (n_neurons, n_timepoints)
            behavior_estimate: Decoded kinematics (n_channels, n_timepoints)
            condition_ids: Estimated conditions (2, n_timepoints)
            state_indices: State indices (2, n_timepoints)
            interpolation_weights: Alpha/beta weights (3, n_timepoints)
        """
        def select_likely_states(conds_to_exclude):
            # Get the maximum likelihood neural state, excluding states that
            # either lack sufficient history or have been explicitly excluded,
            # along with the most likely adjacent state.
            c_hat, k_prime_hats = utils.maximum_likelihood(
                log_posterior.copy(),
                self.settings.history_bins,
                self.state.base_state_indices,
                self.state.lagged_state_indices,
                conds_to_exclude,
            )

            # Convert k_prime_hats to kIdx in a manner that matches how t_prime
            # was converted to tIdx.
            k_idx = utils.get_state_indices(k_prime_hats, time_index_fn, K[c_hat])
            return c_hat, k_prime_hats, k_idx

        def interp_adjacent_states(c_hat, k_prime_hats, kIdx):
            
            # Construct Lambdas
            # Convert condition index to actual condition ID
            cond_id = self.state.condition_list[int(c_hat)]
            idx1 = utils.cond_state_to_flat(
                c_hat,
                k_prime_hats[0] + np.arange(-self.settings.history_bins - 1, 0 + 1),
                self.state.base_state_indices,
            )  # +1 for Python indexing and open brackets
            idx2 = utils.cond_state_to_flat(
                c_hat,
                k_prime_hats[1] + np.arange(-self.settings.history_bins - 1, 0 + 1),
                self.state.base_state_indices,
            )  # +2 for Python indexing and open brackets
            lambda1 = self.settings.rate_bin_centers[self.state.rate_indices[idx1, :] - 1].T  # -1 for Python indexing, squeeze out singleton
            lambda2 = self.settings.rate_bin_centers[self.state.rate_indices[idx2, :] - 1].T  # -1 for Python indexing, squeeze out singleton
            lambda1 = np.squeeze(lambda1, axis=0)
            lambda2 = np.squeeze(lambda2, axis=0)
            # Learn interpolation parameter
            alpha_hat = utils.fit_poisson_interp(
                spike_counts, lambda1, lambda2, self.settings.interp_max_iters, self.settings.interp_tolerance, 0
            )

            # Apply interpolation
            lambda_tilde = (1 - alpha_hat) * lambda1 + alpha_hat * lambda2
            X_hat = (1 - alpha_hat) * self.state.rate_templates[cond_id][
                :, kIdx[0, :]
            ] + alpha_hat * self.state.rate_templates[cond_id][:, kIdx[1, :]]
            Z_hat = (1 - alpha_hat) * self.state.behavior_templates[cond_id][
                :, kIdx[0, :]
            ] + alpha_hat * self.state.behavior_templates[cond_id][:, kIdx[1, :]]
            
            return X_hat, Z_hat, lambda_tilde, alpha_hat

        def interp_states(lambda1, lambda2, X_hat1, X_hat2, Z_hat1, Z_hat2):
            # Learn interpolation parameter
            beta_hat = utils.fit_poisson_interp(
                spike_counts, lambda1, lambda2, self.settings.interp_max_iters, self.settings.interp_tolerance, 0
            )

            # Apply interpolation
            lambda_tilde = (1 - beta_hat) * lambda1 + beta_hat * lambda2
            X_hat = (1 - beta_hat) * X_hat1 + beta_hat * X_hat2
            Z_hat = (1 - beta_hat) * Z_hat1 + beta_hat * Z_hat2

            return X_hat, Z_hat, lambda_tilde, beta_hat

        # Get trajectory length
        K = [phi.shape[1] - 1 for phi in self.state.behavior_templates.values()]  # -1 for Python indexing

        # Proceed depending on interpolation flag
        if self.settings.interp_mode == 0:
            pass

        elif self.settings.interp_mode == 1:
            pass

        elif self.settings.interp_mode == 2:

            # Estimate most likely pair of adjacent states
            # conds_to_exclude = []
            conds_to_exclude = np.array(())
            c_hat_A, k_prime_hats_A, k_idx_a = select_likely_states(conds_to_exclude)

            # Select most likely pair of adjacent states (from different condition)
            conds_to_exclude = [c_hat_A]
            c_hat_B, k_prime_hats_B, k_idx_b = select_likely_states(conds_to_exclude)

            # Interpolate across time
            X_hat_A, Z_hat_A, lambda_tilde_A, alpha_hat_A = interp_adjacent_states(
                c_hat_A, k_prime_hats_A, k_idx_a
            )
            X_hat_B, Z_hat_B, lambda_tilde_B, alpha_hat_B = interp_adjacent_states(
                c_hat_B, k_prime_hats_B, k_idx_b
            )

            # Interpolate across conditions
            X_hat, Z_hat, tilde, beta_hat = interp_states(
                lambda_tilde_A, lambda_tilde_B, X_hat_A, X_hat_B, Z_hat_A, Z_hat_B
            )

            # Store estimated state condition-times
            C_hat = np.array([i * np.ones(Z_hat.shape[1]) for i in [c_hat_A, c_hat_B]])
            K_hat = np.append(k_idx_a, k_idx_b, axis=0)
            alpha_hat = np.array(
                [i * np.ones(Z_hat.shape[1]) for i in [beta_hat, alpha_hat_A, alpha_hat_B]]
            )

        elif self.settings.interp_mode == 3:
            pass

        return X_hat.T, Z_hat.T, C_hat.T, K_hat.T, alpha_hat.T


    def score(
            self,
            spikes_list: List[npt.NDArray],
            truth_list:  List[npt.NDArray],
        ) -> float:
            """
            Compute the coefficient of determination (R²) across trials.

            Parameters
            ----------
            spikes_list : list of ndarray, shape (n_neurons, T)
                Input spike‐count arrays for each trial.
            truth_list : list of ndarray, shape (n_behav_dims, T)
                Ground‐truth behavioral trajectories for each trial.

            Returns
            -------
            r2 : float
                R² score aggregated over all trials.
            """
            # TODO: implement scoring logic
            ...
    
    def save_to_disk(self, file_path: str) -> None:
        """
        Save the decoder's settings and state to disk as JSON.
        
        Parameters
        ----------
        file_path : str
            Path where to save the decoder data
        """
        if self.state is None or not self.state.rate_templates:
            raise ValueError("Cannot save decoder: no state available. Call fit() first.")
            
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            'settings': self._serialize_settings(),
            'state': self._serialize_state()
        }
        with open(file_path, 'w') as f:
            json.dump(save_data, f, indent=2)
    
    def _load_from_disk(self, file_path: str) -> None:
        """
        Load decoder settings and state from disk.
        
        Parameters
        ----------
        file_path : str
            Path to the saved decoder data
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No saved decoder found at: {file_path}")
            
        with open(file_path, 'r') as f:
            save_data = json.load(f)
            
        self.settings = self._deserialize_settings(save_data['settings'])
        self.state = self._deserialize_state(save_data['state'])
    
    def _serialize_settings(self) -> Dict:
        """Convert MINTSettings to JSON-serializable dict."""
        settings_dict = asdict(self.settings)
        
        # Convert numpy arrays and tuples to JSON-serializable types
        for key, value in settings_dict.items():
            if isinstance(value, np.ndarray):
                settings_dict[key] = value.tolist()
            elif isinstance(value, tuple):
                settings_dict[key] = list(value)
                
        return settings_dict
    
    def _deserialize_settings(self, settings_dict: Dict) -> MINTSettings:
        """Convert dict back to MINTSettings object."""
        # Convert lists back to numpy arrays
        array_fields = ['test_alignment', 'trial_alignment', 'trajectories_alignment', 
                       'log_likelihood_lookup', 'rate_bin_centers']
        
        for field in array_fields:
            if field in settings_dict and settings_dict[field] is not None:
                settings_dict[field] = np.array(settings_dict[field])
        
        # Convert lists back to tuples
        if 'rate_range_lookup' in settings_dict and settings_dict['rate_range_lookup'] is not None:
            settings_dict['rate_range_lookup'] = tuple(settings_dict['rate_range_lookup'])
        
        # Remove derived fields that have init=False
        derived_fields = ['rate_range_lookup', 'min_rate_per_bin', 'history_bins', 
                         'log_likelihood_lookup', 'rate_bin_centers']
        constructor_dict = {k: v for k, v in settings_dict.items() if k not in derived_fields}
        
        # Create settings object with only constructor fields
        settings = MINTSettings(**constructor_dict)
        
        # Manually set the derived fields
        for field in derived_fields:
            if field in settings_dict and settings_dict[field] is not None:
                setattr(settings, field, settings_dict[field])
                
        return settings
    
    def _serialize_state(self) -> Dict:
        """Convert MINTState to JSON-serializable dict."""
        if self.state is None:
            return None
            
        state_dict = {
            'rate_templates': {},
            'behavior_templates': {},
            'rate_indices': self.state.rate_indices.tolist() if hasattr(self.state, 'rate_indices') and self.state.rate_indices is not None else None,
            'base_state_indices': self.state.base_state_indices.tolist() if hasattr(self.state, 'base_state_indices') and self.state.base_state_indices is not None else None,
            'lagged_state_indices': self.state.lagged_state_indices.tolist() if hasattr(self.state, 'lagged_state_indices') and self.state.lagged_state_indices is not None else None,
            'shifted_indices_past': self.state.shifted_indices_past.tolist() if hasattr(self.state, 'shifted_indices_past') and self.state.shifted_indices_past is not None else None,
            'shifted_indices_future': self.state.shifted_indices_future.tolist() if hasattr(self.state, 'shifted_indices_future') and self.state.shifted_indices_future is not None else None,
            'interp_map': self.state.interp_map.tolist() if self.state.interp_map is not None else None,
            'condition_list': self.state.condition_list.tolist() if self.state.condition_list is not None else None,
            'kinematic_labels': list(self.state.kinematic_labels)
        }
        
        # Convert template dictionaries
        for cond_id, template in self.state.rate_templates.items():
            state_dict['rate_templates'][str(cond_id)] = template.tolist()
            
        for cond_id, template in self.state.behavior_templates.items():
            state_dict['behavior_templates'][str(cond_id)] = template.tolist()
            
        return state_dict
    
    def _deserialize_state(self, state_dict: Dict) -> MINTState:
        """Convert dict back to MINTState object."""
        if state_dict is None:
            return None
            
        # Convert template dictionaries back
        rate_templates = {}
        for cond_id_str, template_list in state_dict['rate_templates'].items():
            rate_templates[int(cond_id_str)] = np.array(template_list)
            
        behavior_templates = {}
        for cond_id_str, template_list in state_dict['behavior_templates'].items():
            behavior_templates[int(cond_id_str)] = np.array(template_list)
        
        # Convert other arrays back
        kwargs = {
            'rate_templates': rate_templates,
            'behavior_templates': behavior_templates,
            'rate_indices': np.array(state_dict['rate_indices']) if state_dict['rate_indices'] is not None else None,
            'base_state_indices': np.array(state_dict['base_state_indices']) if state_dict['base_state_indices'] is not None else None,
            'lagged_state_indices': np.array(state_dict['lagged_state_indices']) if state_dict['lagged_state_indices'] is not None else None,
            'shifted_indices_past': np.array(state_dict['shifted_indices_past']) if state_dict['shifted_indices_past'] is not None else None,
            'shifted_indices_future': np.array(state_dict['shifted_indices_future']) if state_dict['shifted_indices_future'] is not None else None,
            'interp_map': np.array(state_dict['interp_map']) if state_dict['interp_map'] is not None else None,
            'condition_list': np.array(state_dict['condition_list']) if state_dict['condition_list'] is not None else None,
            'kinematic_labels': tuple(state_dict['kinematic_labels'])
        }
        
        return MINTState(**kwargs)