#!/usr/bin/env python3
"""
Example demonstrating the save/load functionality of the MINT decoder.

This script shows how to:
1. Create and fit a MINT decoder
2. Save the decoder to disk
3. Load the decoder from disk
4. Verify the loaded decoder works identically
"""

import numpy as np
from brn.mint.decoder import MINTDecoder, MINTSettings
from scripts.mc_maze_example.get_trial_data import get_trial_data

def main():
    # Create MINTSettings
    settings = MINTSettings(
        task="mc_maze",
        data_path="./data",  # Placeholder path
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
        sampling_period=0.001,
        soft_norm=5.0,
        min_prob=1e-6,
        min_rate=0.0,
        interp_mode=2,
        interp_max_iters=10,
        interp_tolerance=0.01,
        num_rate_bins=2000
    )
    
    print("Creating and fitting MINT decoder...")
    
    # Create decoder
    decoder = MINTDecoder(settings)
    
    # Load sample data for fitting
    trial_settings = {"trialAlignment": np.arange(-800, 901, 1)}
    spikes, behavior, conditions = get_trial_data(trial_settings, split="train", n_trials=50)
    
    # Fit the decoder
    decoder.fit(spikes, behavior, conditions)
    print(f"Decoder fitted with {len(np.unique(conditions))} conditions")
    
    # Test prediction before saving
    test_spikes = spikes[:5]
    original_rates, original_behavior = decoder.predict(test_spikes)
    print(f"Original prediction shapes: rates={original_rates.shape}, behavior={original_behavior.shape}")
    
    # Save decoder to disk
    save_path = "./examples/saved_decoder.json"
    decoder.save_to_disk(save_path)
    print(f"Decoder saved to {save_path}")
    
    # Load decoder from disk
    print("Loading decoder from disk...")
    loaded_decoder = MINTDecoder(load_from_path=save_path)
    print("Decoder loaded successfully")
    
    # Test prediction with loaded decoder
    loaded_rates, loaded_behavior = loaded_decoder.predict(test_spikes)
    print(f"Loaded prediction shapes: rates={loaded_rates.shape}, behavior={loaded_behavior.shape}")
    
    # Verify predictions are identical
    if np.allclose(original_rates, loaded_rates, equal_nan=True) and \
       np.allclose(original_behavior, loaded_behavior, equal_nan=True):
        print("✓ SUCCESS: Loaded decoder produces identical predictions!")
    else:
        print("✗ ERROR: Predictions differ between original and loaded decoder")
        
    print(f"✓ All tests passed. Save/load functionality is working correctly.")

if __name__ == "__main__":
    main()