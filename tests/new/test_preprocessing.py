import pytest
import numpy as np

# Import the preprocessing functions
from brn.mint.preprocessing import standard_preprocessing, minimal_preprocessing
from scripts.mc_maze_example.get_trial_data import get_trial_data


@pytest.fixture
def sample_data():
    """Create sample data for testing preprocessing functions"""
    # Simplified test data
    n_trials = 10
    n_neurons = 50
    n_kin_vars = 4
    n_timepoints = 1000
    
    # Create synthetic data
    spikes = np.random.poisson(2, (n_trials, n_neurons, n_timepoints))
    behavior = np.random.randn(n_trials, n_kin_vars, n_timepoints) 
    cond_ids = np.random.choice([1, 2, 3], n_trials)
    
    # Time alignments
    trial_alignment = np.arange(-500, 500)  # 1000 timepoints
    trajectories_alignment = np.arange(-300, 200)  # 500 timepoints
    
    return spikes, behavior, cond_ids, trial_alignment, trajectories_alignment


@pytest.fixture
def mc_maze_data():
    """Load real MC Maze data for testing"""
    settings = {"trialAlignment": np.arange(-800, 901, 1)}
    S, Z, condition = get_trial_data(settings, split="train", n_trials=20)
    
    trial_alignment = settings["trialAlignment"]
    trajectories_alignment = np.arange(-500, 701, 1)
    
    return S, Z, condition, trial_alignment, trajectories_alignment


class TestStandardPreprocessing:
    """Test the standard preprocessing function"""
    
    def test_standard_preprocessing_outputs(self, sample_data):
        """Test that standard preprocessing returns correct data types and structures"""
        spikes, behavior, cond_ids, trial_alignment, trajectories_alignment = sample_data
        
        rate_templates, behavior_templates, condition_list = standard_preprocessing(
            spikes=spikes,
            behavior=behavior,
            cond_ids=cond_ids,
            trial_alignment=trial_alignment,
            trajectories_alignment=trajectories_alignment,
            gaussian_sigma=30,
            bin_size=20,
            soft_norm=5.0,
            sampling_period=0.001,
            trial_dims=1,
            neural_dims=np.nan,  # Skip neural PCA
            condition_dims=np.nan,  # Skip condition PCA
        )
        
        # Check return types
        assert isinstance(rate_templates, dict)
        assert isinstance(behavior_templates, dict)
        assert isinstance(condition_list, np.ndarray)
        
        # Check that all unique conditions are present
        unique_conditions = np.unique(cond_ids)
        assert len(rate_templates) == len(unique_conditions)
        assert len(behavior_templates) == len(unique_conditions)
        assert len(condition_list) == len(unique_conditions)
        
        # Check that condition_list matches dict keys
        assert set(rate_templates.keys()) == set(condition_list)
        assert set(behavior_templates.keys()) == set(condition_list)
        
    def test_standard_preprocessing_shapes(self, sample_data):
        """Test that preprocessing produces correct output shapes"""
        spikes, behavior, cond_ids, trial_alignment, trajectories_alignment = sample_data
        
        rate_templates, behavior_templates, condition_list = standard_preprocessing(
            spikes=spikes,
            behavior=behavior,
            cond_ids=cond_ids,
            trial_alignment=trial_alignment,
            trajectories_alignment=trajectories_alignment,
            gaussian_sigma=30,
            bin_size=20,
            soft_norm=5.0,
            sampling_period=0.001,
            trial_dims=1,
            neural_dims=np.nan,
            condition_dims=np.nan,
        )
        
        expected_timepoints = len(trajectories_alignment)
        n_neurons = spikes.shape[1]
        
        # Check shapes
        for cond in condition_list:
            rate_template = rate_templates[cond]
            behavior_template = behavior_templates[cond]
            
            assert rate_template.shape == (n_neurons, expected_timepoints)
            assert behavior_template.shape == (4, expected_timepoints)  # 4 kinematic variables
    
    def test_standard_preprocessing_with_real_data(self, mc_maze_data):
        """Test preprocessing with real MC Maze data"""
        S, Z, condition, trial_alignment, trajectories_alignment = mc_maze_data
        
        rate_templates, behavior_templates, condition_list = standard_preprocessing(
            spikes=S,
            behavior=Z,
            cond_ids=condition,
            trial_alignment=trial_alignment,
            trajectories_alignment=trajectories_alignment,
            gaussian_sigma=30,
            bin_size=20,
            soft_norm=5.0,
            sampling_period=0.001,
            trial_dims=1,
            neural_dims=np.nan,
            condition_dims=21,  # Use condition-level PCA
        )
        
        # Check that we get reasonable outputs
        assert len(rate_templates) > 0
        assert len(behavior_templates) > 0
        assert len(condition_list) == len(np.unique(condition))
        
        # Check that templates are properly shaped
        expected_timepoints = len(trajectories_alignment)
        for cond in condition_list:
            assert rate_templates[cond].shape[1] == expected_timepoints
            assert behavior_templates[cond].shape[1] == expected_timepoints


class TestMinimalPreprocessing:
    """Test the minimal preprocessing function"""
    
    def test_minimal_preprocessing_outputs(self, sample_data):
        """Test that minimal preprocessing returns correct structure"""
        spikes, behavior, cond_ids, trial_alignment, trajectories_alignment = sample_data
        
        rate_templates, behavior_templates, condition_list = minimal_preprocessing(
            spikes=spikes,
            behavior=behavior,
            cond_ids=cond_ids,
            trial_alignment=trial_alignment,
            trajectories_alignment=trajectories_alignment,
        )
        
        # Check return types
        assert isinstance(rate_templates, dict)
        assert isinstance(behavior_templates, dict)
        assert isinstance(condition_list, np.ndarray)
        
        # Check that all unique conditions are present
        unique_conditions = np.unique(cond_ids)
        assert len(rate_templates) == len(unique_conditions)
        assert len(behavior_templates) == len(unique_conditions)
        assert len(condition_list) == len(unique_conditions)
    
    def test_minimal_vs_standard_preprocessing(self, sample_data):
        """Test that minimal preprocessing differs from standard (no smoothing/PCA)"""
        spikes, behavior, cond_ids, trial_alignment, trajectories_alignment = sample_data
        
        # Minimal preprocessing
        rate_min, behavior_min, cond_list_min = minimal_preprocessing(
            spikes, behavior, cond_ids, trial_alignment, trajectories_alignment
        )
        
        # Standard preprocessing  
        rate_std, behavior_std, cond_list_std = standard_preprocessing(
            spikes, behavior, cond_ids, trial_alignment, trajectories_alignment,
            gaussian_sigma=30, bin_size=20, soft_norm=5.0, sampling_period=0.001,
            trial_dims=1, neural_dims=np.nan, condition_dims=np.nan
        )
        
        # Condition lists should be the same
        np.testing.assert_array_equal(cond_list_min, cond_list_std)
        
        # Behavior templates should be the same (both just do trial averaging)
        for cond in cond_list_min:
            np.testing.assert_array_almost_equal(behavior_min[cond], behavior_std[cond])
        
        # Rate templates should be different (standard does smoothing)
        for cond in cond_list_min:
            # They should have same shape but different values due to smoothing
            assert rate_min[cond].shape == rate_std[cond].shape
            # Values should be different due to Gaussian filtering in standard
            assert not np.allclose(rate_min[cond], rate_std[cond])


class TestPreprocessingEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_condition_handling(self):
        """Test preprocessing with minimal data"""
        # Single condition, single trial
        spikes = np.random.poisson(2, (1, 10, 100))
        behavior = np.random.randn(1, 4, 100)
        cond_ids = np.array([1])
        trial_alignment = np.arange(100)
        trajectories_alignment = np.arange(50, 100)
        
        rate_templates, behavior_templates, condition_list = minimal_preprocessing(
            spikes, behavior, cond_ids, trial_alignment, trajectories_alignment
        )
        
        assert len(rate_templates) == 1
        assert len(behavior_templates) == 1
        assert len(condition_list) == 1
        assert condition_list[0] == 1
    
    def test_trajectory_alignment_subset(self, sample_data):
        """Test that trajectory alignment properly subsets the data"""
        spikes, behavior, cond_ids, trial_alignment, trajectories_alignment = sample_data
        
        # Make trajectories_alignment a small subset
        trajectories_alignment = trial_alignment[100:200]  # 100 timepoints
        
        rate_templates, behavior_templates, condition_list = minimal_preprocessing(
            spikes, behavior, cond_ids, trial_alignment, trajectories_alignment
        )
        
        # All templates should have the reduced timepoint count
        for cond in condition_list:
            assert rate_templates[cond].shape[1] == 100
            assert behavior_templates[cond].shape[1] == 100


# Run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v"])