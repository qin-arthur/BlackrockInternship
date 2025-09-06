import pytest
import numpy as np

from brn.mint.decoder import MINTDecoder, MINTSettings, MINTState
from scripts.mc_maze_example.get_trial_data import get_trial_data


@pytest.fixture(scope="session")
def mint_settings():
    """Create MINTSettings for MC Maze testing with optimized parameters"""
    task = "mc_maze"
    data_path = "C:/Users/aqin/Downloads/000128/sub-Jenkins/"
    results_path = "C:/Users/aqin/Downloads/results"
    sampling_period = 0.001
    soft_norm = 5.0
    min_prob = 1e-6
    min_rate = 0.0
    interp_mode = 2
    
    # Optimized time windows for faster testing
    trial_alignment = np.arange(-400, 401, 1)
    test_alignment = np.arange(-250, 251, 1)
    trajectories_alignment = np.arange(-300, 301, 1)
    gaussian_sigma = 30
    neural_dims = np.nan
    condition_dims = 8  # Reduced for smaller test dataset
    trial_dims = 1
    bin_size = 20
    observation_window = 300
    causal = True
    
    min_lambda = 1.0
    interp_max_iters = 10
    interp_tolerance = 0.01
    num_rate_bins = 2000
    
    return MINTSettings(
        task=task,
        data_path=data_path,
        results_path=results_path,
        bin_size=bin_size,
        observation_window=observation_window,
        causal=causal,
        trial_alignment=trial_alignment,
        test_alignment=test_alignment,
        trajectories_alignment=trajectories_alignment,
        gaussian_sigma=gaussian_sigma,
        neural_dims=neural_dims,
        condition_dims=condition_dims,
        trial_dims=trial_dims,
        min_lambda=min_lambda,
        sampling_period=sampling_period,
        soft_norm=soft_norm,
        min_prob=min_prob,
        min_rate=min_rate,
        interp_mode=interp_mode,
        interp_max_iters=interp_max_iters,
        interp_tolerance=interp_tolerance,
        num_rate_bins=num_rate_bins
    )


@pytest.fixture(scope="session")
def trial_data():
    """Load minimal trial data once per test session for fast testing"""
    trial_alignment = np.arange(-400, 401, 1)
    settings = {"trialAlignment": trial_alignment}
    S, Z, condition = get_trial_data(settings, split="train", n_trials=10)
    return S, Z, condition


@pytest.fixture
def empty_state():
    """Create an empty MINTState for initialization"""
    return MINTState(
        rate_templates={},
        behavior_templates={},
        rate_indices=np.array([]),
        base_state_indices=np.array([]),
        lagged_state_indices=np.array([]),
        shifted_indices_past=np.array([]),
        shifted_indices_future=np.array([]),
        interp_map=None,
        condition_list=None,
        kinematic_labels=("xpos", "ypos", "xvel", "yvel")
    )


@pytest.fixture
def decoder(mint_settings, empty_state):
    """Create a MINTDecoder instance"""
    return MINTDecoder(mint_settings, empty_state)


@pytest.fixture(scope="session")
def fitted_decoder(mint_settings, trial_data, tmp_path_factory):
    """Fit decoder once, save to disk, then load for all tests"""
    tmp_dir = tmp_path_factory.mktemp("decoder_cache")
    decoder_path = tmp_dir / "fitted_decoder.json"
    
    # Fit and save decoder
    decoder = MINTDecoder(mint_settings)
    S, Z, condition = trial_data
    fitted = decoder.fit(S, Z, condition)
    fitted.save_to_disk(str(decoder_path))
    
    # Load fresh instance for tests
    loaded_decoder = MINTDecoder(load_from_path=str(decoder_path))
    
    yield loaded_decoder
    
    # Cleanup
    if decoder_path.exists():
        decoder_path.unlink()


@pytest.fixture(scope="session")
def preprocessed_templates(mint_settings, trial_data):
    """Preprocess templates once per test session"""
    from brn.mint.preprocessing import standard_preprocessing
    
    S, Z, condition = trial_data
    return standard_preprocessing(
        spikes=S,
        behavior=Z,
        cond_ids=condition,
        trial_alignment=mint_settings.trial_alignment,
        trajectories_alignment=mint_settings.trajectories_alignment,
        gaussian_sigma=mint_settings.gaussian_sigma,
        bin_size=mint_settings.bin_size,
        soft_norm=mint_settings.soft_norm,
        sampling_period=mint_settings.sampling_period,
        trial_dims=mint_settings.trial_dims,
        neural_dims=mint_settings.neural_dims,
        condition_dims=mint_settings.condition_dims,
    )


# Tests
class TestMINTDecoderFit:
    """Test the fit method of MINTDecoder"""
    
    def test_fit_populates_state(self, decoder, trial_data):
        """Test that fit properly populates all state arrays"""
        S, Z, condition = trial_data
        decoder.fit(S, Z, condition)
        
        n_conditions = len(np.unique(condition))
        n_neurons = S.shape[1]
        
        # Check that all state components are populated
        assert len(decoder.state.rate_templates) == n_conditions
        assert len(decoder.state.behavior_templates) == n_conditions
        
        # Check rate_indices is properly flattened and has correct shape
        assert decoder.state.rate_indices.ndim == 2
        assert decoder.state.rate_indices.shape[1] == n_neurons
        
        # Check index arrays are populated
        assert decoder.state.base_state_indices.size == n_conditions
        assert decoder.state.lagged_state_indices.size > 0
        assert decoder.state.shifted_indices_past.size > 0
        assert decoder.state.shifted_indices_future.size > 0
        
    def test_fit_template_dimensions(self, decoder, trial_data):
        """Test that fitted templates have correct dimensions after smoothing"""
        S, Z, condition = trial_data
        decoder.fit(S, Z, condition)
        
        # Check each condition's templates
        for cond_id, rate_template in decoder.state.rate_templates.items():
            behav_template = decoder.state.behavior_templates[cond_id]
            
            # Rate template should be (n_neurons, n_trajectory_timepoints)
            assert rate_template.shape[0] == S.shape[1]
            
            # Behavior template should be (4, n_trajectory_timepoints)
            assert behav_template.shape[0] == 4
            
            # Both should have same time dimension
            assert rate_template.shape[1] == behav_template.shape[1]


class TestMINTDecoderPredict:
    """Test the predict method of MINTDecoder"""
    
    def test_predict_output_shapes(self, fitted_decoder, trial_data):
        """Test that predict returns correctly shaped outputs"""
        S, Z, condition = trial_data
        
        for n_test_trials in [1, 5, 10]:
            spike_trials = S[:n_test_trials]
            decoded_rates, decoded_behavior = fitted_decoder.predict(spike_trials)
            
            assert decoded_rates.shape == (n_test_trials, S.shape[1], S.shape[2])
            assert decoded_behavior.shape == (n_test_trials, 4, S.shape[2])
    
    def test_predict_rate_constraints(self, fitted_decoder, trial_data):
        """Test that predicted rates respect minimum rate constraint"""
        S, Z, condition = trial_data
        spike_trials = S[:5]
        
        decoded_rates, _ = fitted_decoder.predict(spike_trials)
        
        # Check minimum rate floor is applied
        min_rate_per_bin = fitted_decoder.settings.min_rate_per_bin
        assert np.all(decoded_rates[~np.isnan(decoded_rates)] >= min_rate_per_bin)
    
    def test_predict_causal_masking(self, fitted_decoder, trial_data):
        """Test that causal mode properly masks early timepoints"""
        S, Z, condition = trial_data
        spike_trials = S[:1]  # Just first trial as array
        
        decoded_rates, decoded_behavior = fitted_decoder.predict(spike_trials)
        
        # In causal mode, early timepoints should be NaN
        num_early_invalid = (
            fitted_decoder.settings.bin_size * 
            (fitted_decoder.settings.history_bins + 2) - 1
        )
        
        assert np.all(np.isnan(decoded_rates[0, :, :num_early_invalid]))
        assert np.all(np.isnan(decoded_behavior[0, :, :num_early_invalid]))
        
        # Later timepoints should have valid values
        assert not np.all(np.isnan(decoded_rates[0, :, num_early_invalid:]))
        assert not np.all(np.isnan(decoded_behavior[0, :, num_early_invalid:]))


class TestMINTDecoderEstimate:
    """Test the estimate method functionality"""
    
    def test_estimate_interpolation(self, fitted_decoder):
        """Test that estimate performs interpolation correctly"""
        # Create inputs that would trigger interpolation
        n_states = fitted_decoder.state.rate_indices.shape[0]
        n_neurons = fitted_decoder.state.rate_indices.shape[1]
        
        # Create a log posterior with clear peaks
        log_posterior = -np.ones((n_states, 1)) * 1000  # Very low probability
        log_posterior[100] = -1  # High probability state
        log_posterior[200] = -2  # Another high probability state
        
        # Create spike counts matching the expected window size
        # Window size is history_bins + 2 (based on predict method)
        window_size = fitted_decoder.settings.history_bins + 2
        spike_counts = np.random.poisson(2, (n_neurons, window_size))
        
        # Create a proper time mapping function that returns arrays like get_time_indices
        def time_index_fn(k_prime):
            # Convert scalar to array if needed
            if np.isscalar(k_prime):
                k_prime = np.array([k_prime])
            # Return a realistic range of time indices (e.g., representing a bin_size worth of samples)
            result = np.clip(k_prime, 0, 50).astype(int)
            # For this test, make it return a small range to keep test simple
            if len(result) == 1:
                return np.arange(result[0], result[0] + 20)  # Return 20 timepoints
            return result
        
        # Run estimate
        rate_est, behav_est, cond_ids, state_idx, interp_weights = fitted_decoder.estimate(
            log_posterior, spike_counts, time_index_fn
        )
        
        # Check outputs have consistent dimensions - just verify the basic structure
        n_timepoints = rate_est.shape[0]  # First dimension is timepoints after .T in estimate
        assert rate_est.shape == (n_timepoints, n_neurons)
        assert behav_est.shape == (n_timepoints, 4)
        assert cond_ids.shape[0] == n_timepoints  # First dimension should match
        assert state_idx.shape[0] == n_timepoints  # First dimension should match  
        assert interp_weights.shape[0] == n_timepoints  # First dimension should match
        
        # Check interpolation weights are valid (between 0 and 1)
        assert np.all(interp_weights >= 0)
        assert np.all(interp_weights <= 1)


class TestIntegration:
    """Integration tests for the full pipeline"""
    
    def test_fit_predict_pipeline(self, mint_settings, trial_data):
        """Test the complete fit-predict pipeline"""
        S, Z, condition = trial_data
        
        # Create fresh decoder
        decoder = MINTDecoder(mint_settings, MINTState(
            rate_templates={},
            behavior_templates={},
            rate_indices=np.array([]),
            base_state_indices=np.array([]),
            lagged_state_indices=np.array([]),
            shifted_indices_past=np.array([]),
            shifted_indices_future=np.array([]),
            interp_map=None,
            condition_list=None
        ))
        
        # Split data
        n_train = 40
        train_S, train_Z, train_condition = S[:n_train], Z[:n_train], condition[:n_train]
        test_S = S[n_train:]
        
        # Fit and predict
        decoder.fit(train_S, train_Z, train_condition)
        decoded_rates, decoded_behavior = decoder.predict(test_S)
        
        # Verify outputs are valid
        assert decoded_rates.shape == (10, S.shape[1], S.shape[2])
        assert decoded_behavior.shape == (10, 4, S.shape[2])
        
        # Check that we have some valid (non-NaN) predictions
        valid_mask = ~np.isnan(decoded_rates)
        assert np.sum(valid_mask) > 0
        
        # Check rates are within reasonable bounds
        valid_rates = decoded_rates[valid_mask]
        assert np.all(valid_rates >= 0)
        assert np.all(valid_rates < 1000)  # Reasonable upper bound for spike rates
    
    def test_fit_with_preprocessing_templates(self, mint_settings, trial_data, preprocessed_templates):
        """Test fit method using preprocessed templates instead of raw data"""
        S, Z, condition = trial_data
        rate_templates, behavior_templates, condition_list = preprocessed_templates
        
        # Create fresh decoder
        decoder = MINTDecoder(mint_settings, MINTState(
            rate_templates={}, behavior_templates={}, rate_indices=np.array([]),
            base_state_indices=np.array([]), lagged_state_indices=np.array([]),
            shifted_indices_past=np.array([]), shifted_indices_future=np.array([]),
            interp_map=None, condition_list=None
        ))
        
        # Fit using templates
        decoder.fit(
            rate_templates=rate_templates,
            behavior_templates=behavior_templates,
            condition_list=condition_list
        )
        
        # Verify state is populated
        assert len(decoder.state.rate_templates) == len(np.unique(condition))
        assert len(decoder.state.behavior_templates) == len(np.unique(condition))
        assert decoder.state.condition_list is not None
        
        # Should be able to predict
        test_S = S[:5]
        decoded_rates, decoded_behavior = decoder.predict(test_S)
        assert decoded_rates.shape == (5, S.shape[1], S.shape[2])
        assert decoded_behavior.shape == (5, 4, S.shape[2])
    
    def test_fit_template_vs_raw_consistency(self, mint_settings, trial_data, preprocessed_templates):
        """Test that template-based and raw data fitting produce the same results"""
        S, Z, condition = trial_data
        rate_templates, behavior_templates, condition_list = preprocessed_templates
        
        # Method 1: Raw data fitting
        decoder1 = MINTDecoder(mint_settings, MINTState(
            rate_templates={}, behavior_templates={}, rate_indices=np.array([]),
            base_state_indices=np.array([]), lagged_state_indices=np.array([]),
            shifted_indices_past=np.array([]), shifted_indices_future=np.array([]),
            interp_map=None, condition_list=None
        ))
        decoder1.fit(S, Z, condition)
        
        # Method 2: Template-based fitting using cached preprocessing
        
        decoder2 = MINTDecoder(mint_settings, MINTState(
            rate_templates={}, behavior_templates={}, rate_indices=np.array([]),
            base_state_indices=np.array([]), lagged_state_indices=np.array([]),
            shifted_indices_past=np.array([]), shifted_indices_future=np.array([]),
            interp_map=None, condition_list=None
        ))
        decoder2.fit(
            rate_templates=rate_templates,
            behavior_templates=behavior_templates,
            condition_list=condition_list
        )
        
        # Compare key state components - they should be identical
        np.testing.assert_array_equal(decoder1.state.condition_list, decoder2.state.condition_list)
        
        # Templates should be very close (allowing for small numerical differences)
        for cond in decoder1.state.condition_list:
            np.testing.assert_array_almost_equal(
                decoder1.state.rate_templates[cond], 
                decoder2.state.rate_templates[cond],
                decimal=10
            )
            np.testing.assert_array_almost_equal(
                decoder1.state.behavior_templates[cond],
                decoder2.state.behavior_templates[cond],
                decimal=10
            )
    
    def test_reproducibility(self, mint_settings, trial_data):
        """Test that results are reproducible"""
        S, Z, condition = trial_data
        
        # Create two identical decoders
        state1 = MINTState(
            rate_templates={}, behavior_templates={}, rate_indices=np.array([]),
            base_state_indices=np.array([]), lagged_state_indices=np.array([]),
            shifted_indices_past=np.array([]), shifted_indices_future=np.array([]),
            interp_map=None, condition_list=None
        )
        state2 = MINTState(
            rate_templates={}, behavior_templates={}, rate_indices=np.array([]),
            base_state_indices=np.array([]), lagged_state_indices=np.array([]),
            shifted_indices_past=np.array([]), shifted_indices_future=np.array([]),
            interp_map=None, condition_list=None
        )
        
        decoder1 = MINTDecoder(mint_settings, state1)
        decoder2 = MINTDecoder(mint_settings, state2)
        
        # Fit both with same data
        decoder1.fit(S, Z, condition)
        decoder2.fit(S, Z, condition)
        
        # Predict with same test data
        test_spikes = S[:1]  # First trial as array
        rates1, behavior1 = decoder1.predict(test_spikes)
        rates2, behavior2 = decoder2.predict(test_spikes)
        
        # Results should be identical
        np.testing.assert_array_almost_equal(rates1, rates2)
        np.testing.assert_array_almost_equal(behavior1, behavior2)


class TestMINTDecoderSaveLoad:
    """Test the save/load functionality of MINTDecoder"""
    
    def test_save_load_roundtrip(self, fitted_decoder, tmp_path):
        """Test that a decoder can be saved and loaded with identical state"""
        # Save the fitted decoder
        save_path = tmp_path / "test_decoder.json"
        fitted_decoder.save_to_disk(str(save_path))
        
        # Load the decoder
        loaded_decoder = MINTDecoder(load_from_path=str(save_path))
        
        # Compare settings
        self._compare_settings(fitted_decoder.settings, loaded_decoder.settings)
        
        # Compare state
        self._compare_state(fitted_decoder.state, loaded_decoder.state)
    
    def test_save_unfitted_decoder_raises_error(self, decoder, tmp_path):
        """Test that saving an unfitted decoder raises an error"""
        save_path = tmp_path / "unfitted_decoder.json"
        
        with pytest.raises(ValueError, match="Cannot save decoder: no state available"):
            decoder.save_to_disk(str(save_path))
    
    def test_load_nonexistent_file_raises_error(self):
        """Test that loading from a nonexistent file raises an error"""
        with pytest.raises(FileNotFoundError, match="No saved decoder found at"):
            MINTDecoder(load_from_path="/nonexistent/path.json")
    
    def test_save_creates_directory(self, fitted_decoder, tmp_path):
        """Test that save_to_disk creates the parent directory if it doesn't exist"""
        nested_path = tmp_path / "nested" / "directory" / "decoder.json"
        
        # Directory should not exist yet
        assert not nested_path.parent.exists()
        
        # Save should create the directory
        fitted_decoder.save_to_disk(str(nested_path))
        
        # Directory should now exist and file should be saved
        assert nested_path.parent.exists()
        assert nested_path.exists()
    
    def test_loaded_decoder_predict_equivalence(self, fitted_decoder, trial_data, tmp_path):
        """Test that a loaded decoder produces identical predictions"""
        S, Z, condition = trial_data
        test_spikes = S[:3]  # Use first 3 trials
        
        # Get predictions from original decoder
        orig_rates, orig_behavior = fitted_decoder.predict(test_spikes)
        
        # Save and load decoder
        save_path = tmp_path / "decoder_for_prediction.json"
        fitted_decoder.save_to_disk(str(save_path))
        loaded_decoder = MINTDecoder(load_from_path=str(save_path))
        
        # Get predictions from loaded decoder
        loaded_rates, loaded_behavior = loaded_decoder.predict(test_spikes)
        
        # Predictions should be identical
        np.testing.assert_array_equal(orig_rates, loaded_rates)
        np.testing.assert_array_equal(orig_behavior, loaded_behavior)
    
    def test_loaded_decoder_can_be_refitted(self, fitted_decoder, trial_data, tmp_path):
        """Test that a loaded decoder can be fitted again with new data"""
        S, Z, condition = trial_data
        
        # Save and load decoder
        save_path = tmp_path / "decoder_for_refit.json"
        fitted_decoder.save_to_disk(str(save_path))
        loaded_decoder = MINTDecoder(load_from_path=str(save_path))
        
        # Refit with subset of original data
        subset_S, subset_Z, subset_condition = S[:7], Z[:7], condition[:7]
        loaded_decoder.fit(subset_S, subset_Z, subset_condition)
        
        # Should be able to predict
        test_spikes = S[7:10]
        rates, behavior = loaded_decoder.predict(test_spikes)
        
        assert rates.shape == (3, S.shape[1], S.shape[2])
        assert behavior.shape == (3, 4, S.shape[2])
    
    def test_save_load_with_templates(self, mint_settings, trial_data, tmp_path, preprocessed_templates):
        """Test save/load functionality when decoder was fitted with templates"""
        S, Z, condition = trial_data
        rate_templates, behavior_templates, condition_list = preprocessed_templates
        
        # Fit decoder with templates
        decoder = MINTDecoder(mint_settings)
        decoder.fit(rate_templates=rate_templates, behavior_templates=behavior_templates, condition_list=condition_list)
        
        # Save and load
        save_path = tmp_path / "template_decoder.json"
        decoder.save_to_disk(str(save_path))
        loaded_decoder = MINTDecoder(load_from_path=str(save_path))
        
        # Compare state
        self._compare_state(decoder.state, loaded_decoder.state)
        
        # Test predictions are identical
        test_spikes = S[:2]
        orig_rates, orig_behavior = decoder.predict(test_spikes)
        loaded_rates, loaded_behavior = loaded_decoder.predict(test_spikes)
        
        np.testing.assert_array_equal(orig_rates, loaded_rates)
        np.testing.assert_array_equal(orig_behavior, loaded_behavior)
    
    def _compare_settings(self, settings1, settings2):
        """Helper method to compare two MINTSettings objects"""
        # Compare all fields
        for field_name in settings1.__dataclass_fields__:
            val1 = getattr(settings1, field_name)
            val2 = getattr(settings2, field_name)
            
            if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
                np.testing.assert_array_equal(val1, val2, err_msg=f"Field {field_name} differs")
            elif isinstance(val1, tuple) and isinstance(val2, tuple):
                assert val1 == val2, f"Field {field_name} differs: {val1} != {val2}"
            elif isinstance(val1, (tuple, list)) and isinstance(val2, (tuple, list)):
                # Handle tuple/list comparison (for serialization issues)
                assert list(val1) == list(val2), f"Field {field_name} differs: {val1} != {val2}"
            elif isinstance(val1, float) and isinstance(val2, float) and np.isnan(val1) and np.isnan(val2):
                # Both are NaN, they are equal
                pass
            else:
                assert val1 == val2, f"Field {field_name} differs: {val1} != {val2}"
    
    def _compare_state(self, state1, state2):
        """Helper method to compare two MINTState objects"""
        # Compare template dictionaries
        assert set(state1.rate_templates.keys()) == set(state2.rate_templates.keys())
        assert set(state1.behavior_templates.keys()) == set(state2.behavior_templates.keys())
        
        for cond_id in state1.rate_templates.keys():
            np.testing.assert_array_equal(
                state1.rate_templates[cond_id], 
                state2.rate_templates[cond_id],
                err_msg=f"Rate template for condition {cond_id} differs"
            )
            np.testing.assert_array_equal(
                state1.behavior_templates[cond_id], 
                state2.behavior_templates[cond_id],
                err_msg=f"Behavior template for condition {cond_id} differs"
            )
        
        # Compare array fields
        array_fields = ['rate_indices', 'base_state_indices', 'lagged_state_indices', 
                       'shifted_indices_past', 'shifted_indices_future', 'condition_list']
        
        for field_name in array_fields:
            val1 = getattr(state1, field_name)
            val2 = getattr(state2, field_name)
            
            if val1 is None and val2 is None:
                continue
            elif val1 is None or val2 is None:
                raise AssertionError(f"Field {field_name}: one is None, other is not")
            else:
                np.testing.assert_array_equal(val1, val2, err_msg=f"Field {field_name} differs")
        
        # Compare interp_map (can be None)
        if state1.interp_map is None and state2.interp_map is None:
            pass
        elif state1.interp_map is None or state2.interp_map is None:
            raise AssertionError("interp_map: one is None, other is not")
        else:
            np.testing.assert_array_equal(state1.interp_map, state2.interp_map)
        
        # Compare kinematic_labels
        assert state1.kinematic_labels == state2.kinematic_labels


# Run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v"])