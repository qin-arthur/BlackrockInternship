"""Test movement-based alignment."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from gc_preprocessing import process_single_nwb, create_mint_settings
import numpy as np

# Test on one file
nwb_dir = Path("C:/Users/aqin/Downloads/GeneralizedClick/mint_pipeline_output/nwb_files")
nwb_files = list(nwb_dir.glob("*.nwb"))[:1]

if nwb_files:
    print(f"Testing movement alignment on {nwb_files[0].name}")
    
    # Extract data
    spikes, behavior, condition_ids, trial_data = process_single_nwb(nwb_files[0], verbose=False)
    
    # Use only first 50 trials for quick test
    if len(spikes) > 50:
        spikes = spikes[:50]
        behavior = behavior[:50]
        condition_ids = condition_ids[:50]
        trial_data = trial_data[:50]
        print(f"Limited to first 50 trials for testing")
    
    # Test movement-based settings
    output_path = Path("test_output")
    output_path.mkdir(exist_ok=True)
    
    print("\n=== Testing Movement-Based Alignment ===")
    movement_settings = create_mint_settings(
        train_conds=condition_ids,
        train_spikes=spikes,
        train_behavior=behavior,
        trial_data=trial_data,
        output_path=output_path,
        use_movement_alignment=True
    )
    
    print("\n=== Testing Fixed Alignment ===")
    fixed_settings = create_mint_settings(
        train_conds=condition_ids,
        train_spikes=spikes,
        train_behavior=behavior,
        trial_data=trial_data,
        output_path=output_path,
        use_movement_alignment=False
    )
    
    print("\n=== Comparison ===")
    print(f"Movement-based trial alignment: {len(movement_settings.trial_alignment)} bins")
    print(f"Fixed trial alignment: {len(fixed_settings.trial_alignment)} bins")
    print(f"Movement-based trajectories alignment: {len(movement_settings.trajectories_alignment)} bins")
    print(f"Fixed trajectories alignment: {len(fixed_settings.trajectories_alignment)} bins")
    
else:
    print("No NWB files found")