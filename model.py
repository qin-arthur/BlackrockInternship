import numpy as np
import numpy.typing as npt
import numba as nb
from scipy.stats import poisson
from typing import Tuple, Dict, Callable
from collections import deque
from dataclasses import dataclass



from . import utils


@dataclass
class MINTSettings:
    """Configuration parameters for online MINT decoder.
    
    Attributes:
        task: Task identifier string
        fs: Sampling frequency (Hz)
        obs_window: Observation window size (ms)
        min_lambda: Minimum firing rate threshold
        n_rates: Number of discrete rate bins for quantization
        min_rate: Minimum allowed firing rate (spikes/sec)
        max_rate: Maximum allowed firing rate (spikes/sec)
        min_prob: Minimum probability threshold for Poisson likelihood
        interp_mode: Interpolation strategy (0=none, 1=time, 2=condition, 3=both)
        interp_max_iters: Maximum iterations for interpolation optimization
        interp_tolerance: Convergence tolerance for interpolation
    """
    task: str
    fs: float
    obs_window: int
    min_lambda: float
    n_rates: int
    min_rate: int
    max_rate: int
    min_prob: float
    interp_mode: int
    interp_max_iters: int
    interp_tolerance: float

    
class MINT:
    """Online MINT decoder for trial-by-trial neural decoding.
    
    Designed for real-time decoding with binned neural and behavioral data.
    All input data must be pre-binned and preprocessed upstream.
    """
    
    def __init__(self, settings: MINTSettings | None = None, filepath: str | None = None, **kwargs):
        """Initialize MINT decoder with settings or load from file.
        
        Args:
            settings: Configuration parameters
            filepath: Path to saved decoder state
            **kwargs: Alternative way to specify settings parameters
        """
        if settings is None and filepath is None:
            settings = MINTSettings(**kwargs)
        elif settings is None and filepath is not None:
            settings = 0

        # TODO: Store settings and state to member variables
        self.settings = settings
        
        # Compute derived settings
        if filepath is None:
            dt = 1/self.settings.fs
            # Miscellaneous
            lambda_range = [x * dt for x in [self.settings.min_lambda, 500]]
            
            # Corresponds to tau_prime in the paper
            n_history_bins = (
                round(self.settings.obs_window / 20) - 2
            )
            
            min_spikes = self.settings.min_rate * dt
            
            # Precompute lookup table of log-likelihoods
            rates = np.linspace(lambda_range[0], lambda_range[1], self.settings.n_rates)
            rates = np.array(rates).reshape(self.settings.n_rates, 1)
            
            max_spikes = round(dt * self.settings.max_rate)

            counts_mat = np.matlib.repmat(np.arange(0, max_spikes + 1), self.settings.n_rates, 1)
            rates_mat  = np.matlib.repmat(rates, 1, max_spikes + 1)

            L = poisson.pmf(counts_mat, rates_mat)
            L[L <= self.settings.min_prob] = np.nan
            norm = 1 - self.settings.min_prob * np.sum(np.isnan(L), axis=1)
            L = L * np.array(norm / np.nansum(L, axis=1)).reshape(self.settings.n_rates, 1)
            L[np.isnan(L)] = self.settings.min_prob
            L = np.log(L)
            
        else:
            # Placeholder for loading from file
            self.cond_list = 0
            self.training_trajs = 0
            self.behavior_trajs = 0
            self.rate_indices = 0
            self.first_idx = 0
            self.first_tau_prime_idx = 0
            self.shifted_idx1 = 0
            self.shifted_idx2 = 0
            
            lambda_range = 0
            rates = 0
            L = 0
            n_history_bins = 0
            min_spikes = 0
        # Declaration of variables needed for training and/or inference
        self.lambda_range = lambda_range
        self.rates = rates
        self.L = L
        self.n_history_bins = n_history_bins
        self.min_spikes = min_spikes
        
        
        


    def fit(self, features, behavior, conditions) -> "MINT":
        """Train decoder on preprocessed neural and behavioral data.
        
        Expects fully preprocessed data (binned, smoothed, aligned).
        Builds rate templates and state indices for online decoding.
        
        Args:
            features: Neural data (n_trials, n_time_bins, n_neurons)
            behavior: Kinematic data (n_trials, n_time_bins, n_channels)
            conditions: Trial condition labels (n_trials,)
            
        Returns:
            Self for method chaining
        """
                # Step 1: Reorganize data dimensions - swap time and neuron axes
        features = features.swapaxes(1, 2)  # Now (n_trials, n_neurons, n_time_bins)
        behavior = behavior.swapaxes(1, 2)  # Now (n_trials, n_channels, n_time_bins)
        
        # Step 2: Extract unique conditions from data
        self.cond_list = np.unique(conditions)
        
        # Step 3: Group neural data by condition (Omega_plus in MINT paper)
        self.training_trajs = {cond: features[conditions == cond] for cond in self.cond_list}
        
        # Step 4: Group and average behavior by condition (Phi_plus in MINT paper)
        bsort = {cond: behavior[conditions == cond] for cond in self.cond_list}
        self.behavior_trajs = {k: np.mean(v, axis=0) for k, v in bsort.items()}  # Trial-averaged kinematics

        # Step 5: Average neural trajectories across trials within each condition
        self.training_trajs = {k: np.mean(v, axis=0) for k, v in self.training_trajs.items()}
        
        # Step 6: Quantize firing rates to discrete bins (V in MINT paper)
        # Maps continuous rates to discrete indices for efficient lookup
        self.rate_indices = {
            c: utils.get_rate_indices(x, self.lambda_range, self.settings.n_rates)
            for c, x in self.training_trajs.items()
        }

        # Step 7: Calculate cumulative indices for state-space organization
        rate_indice_shapes = [v.shape[1] for v in self.rate_indices.values()]
        self.first_idx = np.cumsum([0] + rate_indice_shapes[:-1])  # Start index for each condition
        last_idx = np.cumsum(rate_indice_shapes)  # End index for each condition
        
        # Step 8: Build indices for history window (tau_prime)
        tmp = (np.arange(self.n_history_bins)[:, None] + 
               self.first_idx[None, :]).flatten()
        self.first_tau_prime_idx = np.sort(tmp)  # Sorted indices for history recursion
        
        # Step 9: Flatten rate indices into single array for efficient access
        n_neurons = self.rate_indices[list(self.rate_indices.keys())[0]].shape[0]
        self.rate_indices = np.stack([v.T for v in self.rate_indices.values()], 
            axis=0).reshape((-1, n_neurons)  # Shape: (total_states, n_neurons)
        )
        
        # Step 10: Create shifted indices for past/future state lookups
        # Used in recursion to efficiently access lagged states
        self.shifted_idx1 = np.hstack([  # Past window indices
            np.arange(
                self.first_idx[_],
                last_idx[_] - self.n_history_bins
            )
            for _ in range(len(self.training_trajs))
        ])
        self.shifted_idx2 = np.hstack([  # Future window indices
            np.arange(
                self.first_idx[_] + self.n_history_bins,
                last_idx[_]
            )
            for _ in range(len(self.training_trajs))
        ])        
        # Step 11: Initialize decoder state for inference
        self.clear_history()

        return self
    
    
    def fit_mc(self, features, behavior, conditions) -> "MINT":
        """Alternative fit method expecting pre-organized dictionary data.
        
        Designed for mc_maze dataset where data is already in dictionary format.
        
        Args:
            features: Dict of neural templates by condition
            behavior: Dict of behavior templates by condition  
            conditions: List of condition identifiers
            
        Returns:
            Self for method chaining
        """
                # Store pre-binned templates directly
        self.training_trajs = features  # Neural templates (already binned)
        self.behavior_trajs = behavior  # Behavior templates (already binned)  
        self.cond_list = conditions
        
        # Calculate rate indices from binned neural data
        self.rate_indices = {
            c: utils.get_rate_indices(x, self.lambda_range, self.settings.n_rates)
            for c, x in self.training_trajs.items()
        }

        # Step 7: Calculate cumulative indices for state-space organization
        rate_indice_shapes = [v.shape[1] for v in self.rate_indices.values()]
        self.first_idx = np.cumsum([0] + rate_indice_shapes[:-1])  # Start index for each condition
        last_idx = np.cumsum(rate_indice_shapes)  # End index for each condition
        
        # Step 8: Build indices for history window (tau_prime)
        tmp = (np.arange(self.n_history_bins)[:, None] + 
               self.first_idx[None, :]).flatten()
        self.first_tau_prime_idx = np.sort(tmp)  # Sorted indices for history recursion
        
        # Step 9: Flatten rate indices into single array for efficient access
        n_neurons = self.rate_indices[list(self.rate_indices.keys())[0]].shape[0]
        self.rate_indices = np.stack([v.T for v in self.rate_indices.values()], 
            axis=0).reshape((-1, n_neurons)  # Shape: (total_states, n_neurons)
        )
        
        # Step 10: Create shifted indices for past/future state lookups
        # Used in recursion to efficiently access lagged states
        self.shifted_idx1 = np.hstack([  # Past window indices
            np.arange(
                self.first_idx[_],
                last_idx[_] - self.n_history_bins
            )
            for _ in range(len(self.training_trajs))
        ])
        self.shifted_idx2 = np.hstack([  # Future window indices
            np.arange(
                self.first_idx[_] + self.n_history_bins,
                last_idx[_]
            )
            for _ in range(len(self.training_trajs))
        ])        
        # Step 11: Initialize decoder state for inference
        self.clear_history()

        return self

    def clear_history(self):
        """Reset decoder state for new trial/segment."""
        self.Q = np.zeros((self.rate_indices.shape[0], 1))
        
    def predict(self, features: npt.NDArray, new_segment: bool = False):
        """Decode neural and behavioral states from binned spike data.
        
        Performs online bin-by-bin decoding using recursive Bayesian inference.
        
        Args:
            features: Binned spike counts (n_time_bins, n_neurons)
            new_segment: Whether to reset decoder state
            
        Returns:
            X_hat: Decoded firing rates (n_time_bins, n_neurons)
            Z_hat: Decoded kinematics (n_time_bins, n_channels)
            C_hat: Condition estimates
            K_hat: State index estimates
            Alpha_hat: Interpolation weights
        """
        if new_segment:
            self.clear_history()
            
        # Transpose to (n_neurons, n_time_bins) for internal processing
        features = features.T
        n_neurons, n_bins = features.shape
        
        # Initialize output arrays
        X_hat = np.zeros(features.shape)
        kin_dim = next(iter(self.behavior_trajs.values())).shape[0]
        Z_hat = np.zeros((kin_dim, n_bins))
        C_hat, K_hat, Alpha_hat = [], [], []
        
        # Process each time bin sequentially
        for t in range(n_bins):
            # Extract current bin's spike counts
            s_new = features[:, t]
            
            # Get lagged spike counts for history window
            if t > (self.n_history_bins + 1):
                s_old = features[:, t - self.n_history_bins - 2]
            else:
                s_old = np.zeros_like(s_new)  # No history at start

            # Update log-posterior using recursive Bayesian inference
            self.Q = utils.recursion_jit(
                self.Q, s_new, s_old,
                t,
                self.L,
                self.rate_indices,
                self.first_idx,
                self.shifted_idx1,
                self.shifted_idx2,
                n_neurons,
                self.n_history_bins
            )
            
            # Maintain 2D shape for posterior
            if self.Q.ndim == 1:
                self.Q = self.Q.reshape(-1, 1)

            # Decode once sufficient history accumulated
            if t > self.n_history_bins:
                # Extract observation window for current estimate
                S_curr = np.double(features[:, t - self.n_history_bins - 1 : t + 1])
                
                # Perform state estimation using current posterior and observations
                Xb, Zb, Cb, Kb, Ab = self.estimate(
                    self.Q, S_curr
                )
                
                # Extract single-bin output for online decoding
                X_hat[:, t] = Xb[0, :] if Xb.ndim > 1 else Xb
                Z_hat[:, t] = Zb[0, :] if Zb.ndim > 1 else Zb
                C_hat.append(Cb[0, :] if Cb.ndim > 1 else Cb)
                K_hat.append(Kb[0, :] if Kb.ndim > 1 else Kb)
                Alpha_hat.append(Ab[0, :] if Ab.ndim > 1 else Ab)
                
        # Apply minimum firing rate constraint
        np.clip(X_hat, self.min_spikes, np.inf, out=X_hat)
        
        # Transpose back to original dimensions
        return X_hat.T, Z_hat.T, C_hat, K_hat, Alpha_hat
    
        
    def estimate(
        self,
        log_posterior,
        spike_counts,
    ):
        """Estimate neural and behavioral states from posterior distribution.
        
        Selects most likely states and performs interpolation based on mode.
        
        Args:
            log_posterior: Current log-posterior over states (n_states, 1)
            spike_counts: Observation window (n_neurons, window_bins)
            
        Returns:
            rate_estimate: Decoded firing rates (n_neurons, n_timepoints)
            behavior_estimate: Decoded kinematics (n_channels, n_timepoints)
            condition_ids: Estimated conditions (2, n_timepoints)
            state_indices: State indices (2, n_timepoints)
            interpolation_weights: Alpha/beta weights (3, n_timepoints)
        """
        def select_likely_states(conds_to_exclude):
            """Find most likely state pair from posterior."""
            c_hat, k_prime_hats = utils.maximum_likelihood(
                log_posterior.copy(),
                self.n_history_bins,
                self.first_idx,
                self.first_tau_prime_idx,
                conds_to_exclude,
            )

            # k_prime_hats are already bin indices (bin-space operation)
            k_idx = np.array([k_prime_hats[0], k_prime_hats[1]], dtype=int)
            k_idx = np.clip(k_idx, 0, K[c_hat])
            k_idx = k_idx.reshape(2, 1)  # shape (2, 1) for single bin
            return c_hat, k_prime_hats, k_idx

        def interp_adjacent_states(c_hat, k_prime_hats, kIdx):
            """Interpolate between adjacent states within a condition."""
            cond_id = self.cond_list[int(c_hat)]
            idx1 = utils.cond_state_to_flat(
                c_hat,
                k_prime_hats[0] + np.arange(-self.n_history_bins - 1, 0 + 1),
                self.first_idx,
            )  # +1 for Python indexing and open brackets
            idx2 = utils.cond_state_to_flat(
                c_hat,
                k_prime_hats[1] + np.arange(-self.n_history_bins - 1, 0 + 1),
                self.first_idx,
            )  # +2 for Python indexing and open brackets
            lambda1 = self.rates[self.rate_indices[idx1, :] - 1].T  # -1 for Python indexing, squeeze out singleton
            lambda2 = self.rates[self.rate_indices[idx2, :] - 1].T  # -1 for Python indexing, squeeze out singleton
            lambda1 = np.squeeze(lambda1, axis=0)
            lambda2 = np.squeeze(lambda2, axis=0)
            
            # Learn interpolation parameter
            alpha_hat = utils.fit_poisson_interp(
                spike_counts, lambda1, lambda2, self.settings.interp_max_iters, self.settings.interp_tolerance, 0
            )

            # Apply interpolation
            lambda_tilde = (1 - alpha_hat) * lambda1 + alpha_hat * lambda2
            
            # Extract scalar indices for single bin
            k1, k2 = int(kIdx[0, 0]), int(kIdx[1, 0])
            
            # Ensure indices are within bounds
            template_shape = self.behavior_trajs[cond_id].shape
            if k1 >= template_shape[1] or k2 >= template_shape[1]:
                k1 = min(k1, template_shape[1] - 1)
                k2 = min(k2, template_shape[1] - 1)
            
            # Use pre-binned templates directly for interpolation
            X_hat = (1 - alpha_hat) * self.training_trajs[cond_id][:, k1:k1+1] + \
                    alpha_hat * self.training_trajs[cond_id][:, k2:k2+1]
            Z_hat = (1 - alpha_hat) * self.behavior_trajs[cond_id][:, k1:k1+1] + \
                    alpha_hat * self.behavior_trajs[cond_id][:, k2:k2+1]

            return X_hat, Z_hat, lambda_tilde, alpha_hat

        def interp_states(lambda1, lambda2, X_hat1, X_hat2, Z_hat1, Z_hat2):
            """Interpolate between states from different conditions."""
            beta_hat = utils.fit_poisson_interp(
                spike_counts, lambda1, lambda2, self.settings.interp_max_iters, self.settings.interp_tolerance, 0
            )

            # Apply interpolation
            lambda_tilde = (1 - beta_hat) * lambda1 + beta_hat * lambda2
            X_hat = (1 - beta_hat) * X_hat1 + beta_hat * X_hat2
            Z_hat = (1 - beta_hat) * Z_hat1 + beta_hat * Z_hat2

            return X_hat, Z_hat, lambda_tilde, beta_hat

        # Get trajectory lengths for each condition
        K = [phi.shape[1] - 1 for phi in self.behavior_trajs.values()]

        # Select interpolation strategy based on mode
        if self.settings.interp_mode == 0:
            conds_to_exclude = np.array(())
            c_hat, _, k_idx = select_likely_states(conds_to_exclude)
            # pick the “first” of the two returned adjacent states
            cond_id = self.cond_list[int(c_hat)]
            k0 = int(k_idx[0, 0])
            X_hat = self.training_trajs[cond_id][:, k0:k0+1]
            Z_hat = self.behavior_trajs[cond_id][:, k0:k0+1]
            C_hat = np.full((1, Z_hat.shape[1]), c_hat, dtype=int)
            K_hat = k_idx[0:1, :]
            Alpha_hat = np.full((1, Z_hat.shape[1]), np.nan)

        # MODE 1: Time interpolation within condition
        elif self.settings.interp_mode == 1:
            conds_to_exclude = np.array(())
            c_hat, k_prime_hats, k_idx = select_likely_states(conds_to_exclude)
            X_hat, Z_hat, _, alpha_hat = interp_adjacent_states(c_hat, k_prime_hats, k_idx)
            C_hat = np.full((1, Z_hat.shape[1]), np.nan)
            K_hat = np.full((1, Z_hat.shape[1]), np.nan)
            # alpha_hat is scalar in bin-space, expand to match output dims
            Alpha_hat = np.full((1, Z_hat.shape[1]), alpha_hat)
            
        # MODE 2: Cross-condition interpolation only
        elif self.settings.interp_mode == 2:
            conds_to_exclude = np.array(())
            cA, kA_primes, kA_idx = select_likely_states(conds_to_exclude)
            conds_to_exclude = [cA]
            cB, kB_primes, kB_idx = select_likely_states(conds_to_exclude)

            # Get raw templates for each condition
            condA_id = self.cond_list[int(cA)]
            condB_id = self.cond_list[int(cB)]
            kA0 = int(kA_idx[0, 0])
            kB0 = int(kB_idx[0, 0])
            X1 = self.training_trajs[condA_id][:, kA0:kA0+1]
            Z1 = self.behavior_trajs[condA_id][:, kA0:kA0+1]
            X2 = self.training_trajs[condB_id][:, kB0:kB0+1]
            Z2 = self.behavior_trajs[condB_id][:, kB0:kB0+1]

            # Get lambda vectors for cross-condition interpolation
            idxA = utils.cond_state_to_flat(cA, kA0 + np.arange(-self.n_history_bins - 1, 0 + 1), self.first_idx)
            idxB = utils.cond_state_to_flat(cB, kB0 + np.arange(-self.n_history_bins - 1, 0 + 1), self.first_idx)
            λ1 = self.rates[self.rate_indices[idxA, :] - 1].T
            λ2 = self.rates[self.rate_indices[idxB, :] - 1].T
            λ1 = np.squeeze(λ1, axis=0)
            λ2 = np.squeeze(λ2, axis=0)
            X_hat, Z_hat, _, beta_hat = interp_states(λ1, λ2, X1, X2, Z1, Z2)

            C_hat = np.vstack([
                np.full(Z_hat.shape[1], cA),
                np.full(Z_hat.shape[1], cB),
            ])
            K_hat = np.vstack([kA_idx[0, :], kB_idx[0, :]])
            Alpha_hat = np.full((1, Z_hat.shape[1]), beta_hat)


        # MODE 3: Time and condition interpolation
        elif self.settings.interp_mode == 3:
            # Find most likely state pair
            conds_to_exclude = np.array(())
            c_hat_A, k_prime_hats_A, k_idx_a = select_likely_states(conds_to_exclude)

            # Find state pair from different condition
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

            # Store state estimates
            C_hat = np.array([i * np.ones(Z_hat.shape[1]) for i in [c_hat_A, c_hat_B]])
            K_hat = np.append(k_idx_a, k_idx_b, axis=0)
            Alpha_hat = np.array(
                [i * np.ones(Z_hat.shape[1]) for i in [beta_hat, alpha_hat_A, alpha_hat_B]]
            )
        
        else:
            raise ValueError(f"Unknown interp_mode {self.settings.interp_mode}")


        return X_hat.T, Z_hat.T, C_hat.T, K_hat.T, Alpha_hat.T