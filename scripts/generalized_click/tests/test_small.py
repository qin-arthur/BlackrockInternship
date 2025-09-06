"""Test script with just first file and limited trials."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from gc_preprocessing import process_single_nwb, split_train_test, create_mint_settings
from brn.mint.decoder import MINTDecoder
import numpy as np

# Test on one file
nwb_dir = Path("C:/Users/aqin/Downloads/GeneralizedClick/mint_pipeline_output/nwb_files")
nwb_files = list(nwb_dir.glob("*.nwb"))[:1]  # Just first file

if nwb_files:
    print(f"Testing on {nwb_files[0].name}")
    
    # Extract data
    spikes, behavior, condition_ids, trial_data = process_single_nwb(nwb_files[0], verbose=True)
    
    # Use only first 100 trials for quick test
    if len(spikes) > 100:
        spikes = spikes[:100]
        behavior = behavior[:100]
        condition_ids = condition_ids[:100]
        trial_data = trial_data[:100]
        print(f"\nLimited to first 100 trials for testing")
    
    # Split data
    train_spikes, train_behavior, train_conds, test_spikes, test_behavior, test_conds = split_train_test(
        spikes, behavior, condition_ids, test_fraction=0.2, verbose=True
    )
    
    # Create settings
    output_path = Path("test_output")
    output_path.mkdir(exist_ok=True)
    
    settings = create_mint_settings(
        train_conds=train_conds,
        train_spikes=train_spikes,
        train_behavior=train_behavior,
        trial_data=trial_data,
        output_path=output_path
    )
    
    # Initialize and train decoder
    print("\nInitializing MINT decoder...")
    decoder = MINTDecoder(settings)
    
    print("Training decoder...")
    decoder.fit(
        spikes=train_spikes,
        behavior=train_behavior,
        cond_ids=train_conds
    )
    
    print("Training complete!")
    
    # Test
    print("Testing decoder...")
    X_hat, Z_hat = decoder.predict(test_spikes)
    
    print(f"Predictions shape: Neural {X_hat.shape}, Behavior {Z_hat.shape}")
    print("Test complete!")
else:
    print("No NWB files found")