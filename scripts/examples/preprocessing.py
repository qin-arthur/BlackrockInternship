"""
Example: Using MINT with Optional Preprocessing

This example demonstrates the two ways to use the MINT decoder:
1. Automatic preprocessing with raw data (for mc_maze, area2_bump tasks)
2. Manual preprocessing for custom workflows (e.g., mc_rtt with LFADS)
"""

import numpy as np
from brn.mint.decoder import MINTDecoder, MINTSettings, MINTState
from brn.mint.preprocessing import standard_preprocessing, minimal_preprocessing


def create_sample_settings():
    """Create sample MINT settings for demonstration"""
    return MINTSettings(
        task="mc_maze",
        data_path="./data",
        results_path="./results", 
        bin_size=20,
        observation_window=300,
        causal=True,
        trial_alignment=np.arange(-800, 901, 1),
        test_alignment=np.arange(-250, 451, 1),
        trajectories_alignment=np.arange(-500, 701, 1),
        gaussian_sigma=30,
        neural_dims=np.nan,
        condition_dims=21,
        trial_dims=1,
        min_lambda=1.0,
    )


def create_sample_data():
    """Create synthetic data for demonstration"""
    n_trials = 50
    n_neurons = 100  
    n_timepoints = 1701  # Length of trial_alignment
    n_kin_vars = 4
    
    # Synthetic spike and behavior data
    spikes = np.random.poisson(3, (n_trials, n_neurons, n_timepoints))
    behavior = np.random.randn(n_trials, n_kin_vars, n_timepoints)
    
    # Add some position offset to simulate movement onset alignment
    behavior[:, :2, :] += np.cumsum(behavior[:, 2:, :], axis=2) * 0.001
    
    # Random condition assignment
    conditions = np.random.choice([1, 2, 3, 4, 5], n_trials)
    
    return spikes, behavior, conditions


def example_1_automatic_preprocessing():
    """
    Example 1: Automatic preprocessing (recommended for mc_maze, area2_bump)
    
    This is the simplest approach - just provide raw data and let MINT
    handle all preprocessing automatically.
    """
    print("Example 1: Automatic preprocessing")
    print("=" * 50)
    
    # Create settings and data
    settings = create_sample_settings()
    spikes, behavior, conditions = create_sample_data()
    
    # Create decoder with empty state
    decoder = MINTDecoder(settings, MINTState(
        rate_templates={}, behavior_templates={}, rate_indices=np.array([]),
        base_state_indices=np.array([]), lagged_state_indices=np.array([]),
        shifted_indices_past=np.array([]), shifted_indices_future=np.array([]),
        interp_map=None, condition_list=None
    ))
    
    # Fit using raw data - preprocessing happens automatically
    print("Fitting decoder with automatic preprocessing...")
    decoder.fit(spikes=spikes, behavior=behavior, cond_ids=conditions)
    
    print(f"Fitted decoder with {len(decoder.state.rate_templates)} conditions")
    print(f"Rate template shapes: {[v.shape for v in decoder.state.rate_templates.values()][:3]}...")
    
    # Test prediction
    test_spikes = spikes[:5]  # First 5 trials
    decoded_rates, decoded_behavior = decoder.predict(test_spikes)
    print(f"Decoded rates shape: {decoded_rates.shape}")
    print(f"Decoded behavior shape: {decoded_behavior.shape}")
    print()


def example_2_manual_preprocessing():
    """
    Example 2: Manual preprocessing (for custom workflows)
    
    This approach gives you full control over preprocessing steps.
    Useful for mc_rtt with LFADS data or custom preprocessing pipelines.
    """
    print("Example 2: Manual preprocessing")
    print("=" * 50)
    
    # Create settings and data
    settings = create_sample_settings()
    spikes, behavior, conditions = create_sample_data()
    
    # Option A: Use standard preprocessing function
    print("Option A: Using standard_preprocessing() function")
    rate_templates, behavior_templates, condition_list = standard_preprocessing(
        spikes=spikes,
        behavior=behavior,
        cond_ids=conditions,
        trial_alignment=settings.trial_alignment,
        trajectories_alignment=settings.trajectories_alignment,
        gaussian_sigma=settings.gaussian_sigma,
        bin_size=settings.bin_size,
        soft_norm=settings.soft_norm,
        sampling_period=settings.sampling_period,
        trial_dims=settings.trial_dims,
        neural_dims=settings.neural_dims,
        condition_dims=settings.condition_dims,
    )
    
    # Create decoder and fit with preprocessed templates
    decoder_a = MINTDecoder(settings, MINTState(
        rate_templates={}, behavior_templates={}, rate_indices=np.array([]),
        base_state_indices=np.array([]), lagged_state_indices=np.array([]),
        shifted_indices_past=np.array([]), shifted_indices_future=np.array([]),
        interp_map=None, condition_list=None
    ))
    
    decoder_a.fit(
        rate_templates=rate_templates,
        behavior_templates=behavior_templates, 
        condition_list=condition_list
    )
    print(f"  Fitted with {len(rate_templates)} conditions")
    
    # Option B: Use minimal preprocessing (for pre-smoothed data like LFADS)
    print("\\nOption B: Using minimal_preprocessing() for pre-smoothed data")
    rate_templates_min, behavior_templates_min, condition_list_min = minimal_preprocessing(
        spikes=spikes,  # Assume this is already smoothed (e.g., LFADS latents)
        behavior=behavior,
        cond_ids=conditions,
        trial_alignment=settings.trial_alignment,
        trajectories_alignment=settings.trajectories_alignment,
    )
    
    decoder_b = MINTDecoder(settings, MINTState(
        rate_templates={}, behavior_templates={}, rate_indices=np.array([]),
        base_state_indices=np.array([]), lagged_state_indices=np.array([]),
        shifted_indices_past=np.array([]), shifted_indices_future=np.array([]),
        interp_map=None, condition_list=None
    ))
    
    decoder_b.fit(
        rate_templates=rate_templates_min,
        behavior_templates=behavior_templates_min,
        condition_list=condition_list_min
    )
    print(f"  Fitted with {len(rate_templates_min)} conditions")
    print()


def example_3_custom_preprocessing():
    """
    Example 3: Fully custom preprocessing
    
    Shows how to implement your own preprocessing pipeline while
    still using the template-based fit interface.
    """
    print("Example 3: Custom preprocessing pipeline")
    print("=" * 50)
    
    settings = create_sample_settings()
    spikes, behavior, conditions = create_sample_data()
    
    # Custom preprocessing pipeline
    print("Applying custom preprocessing...")
    
    # Step 1: Custom smoothing (e.g., exponential smoothing instead of Gaussian)
    def exponential_smooth(data, alpha=0.1):
        """Simple exponential smoothing"""
        smoothed = np.zeros_like(data)
        smoothed[:, :, 0] = data[:, :, 0]
        for t in range(1, data.shape[2]):
            smoothed[:, :, t] = alpha * data[:, :, t] + (1 - alpha) * smoothed[:, :, t-1]
        return smoothed
    
    spikes_smooth = exponential_smooth(spikes.astype(float))
    
    # Step 2: Custom behavior processing (e.g., velocity-based alignment)  
    behavior_processed = behavior.copy()
    # Could add custom kinematics processing here
    
    # Step 3: Time alignment
    trial_alignment = settings.trial_alignment
    trajectories_alignment = settings.trajectories_alignment
    t_mask = np.isin(trial_alignment, trajectories_alignment)
    spikes_smooth = spikes_smooth[:, :, t_mask]
    behavior_processed = behavior_processed[:, :, t_mask]
    
    # Step 4: Group by conditions and compute templates
    condition_list = np.unique(conditions)
    rate_templates = {}
    behavior_templates = {}
    
    for cond in condition_list:
        cond_mask = conditions == cond
        # Custom averaging (e.g., median instead of mean)
        rate_templates[cond] = np.median(spikes_smooth[cond_mask], axis=0)
        behavior_templates[cond] = np.mean(behavior_processed[cond_mask], axis=0)
    
    print(f"Created {len(rate_templates)} custom templates")
    
    # Fit decoder with custom templates
    decoder = MINTDecoder(settings, MINTState(
        rate_templates={}, behavior_templates={}, rate_indices=np.array([]),
        base_state_indices=np.array([]), lagged_state_indices=np.array([]),
        shifted_indices_past=np.array([]), shifted_indices_future=np.array([]),
        interp_map=None, condition_list=None
    ))
    
    decoder.fit(
        rate_templates=rate_templates,
        behavior_templates=behavior_templates,
        condition_list=condition_list
    )
    
    print(f"Custom decoder fitted successfully!")
    print()


def main():
    """Run all examples"""
    print("MINT Preprocessing Examples")
    print("=" * 70)
    print()
    
    try:
        example_1_automatic_preprocessing()
        example_2_manual_preprocessing() 
        example_3_custom_preprocessing()
        
        print("All examples completed successfully!")
        print()
        print("Summary:")
        print("- Use automatic preprocessing for standard tasks (mc_maze, area2_bump)")
        print("- Use standard_preprocessing() function for more control")
        print("- Use minimal_preprocessing() for pre-smoothed data (LFADS, etc.)")
        print("- Implement custom preprocessing for specialized workflows")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()