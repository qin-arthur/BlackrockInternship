"""Test script to verify trial extraction is working."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from gc_preprocessing import process_single_nwb

# Test on one file
nwb_dir = Path("C:/Users/aqin/Downloads/GeneralizedClick/mint_pipeline_output/nwb_files")
nwb_files = list(nwb_dir.glob("*.nwb"))

if nwb_files:
    print(f"Testing extraction on {nwb_files[0].name}")
    try:
        spikes, behavior, condition_ids, trial_data = process_single_nwb(nwb_files[0], verbose=True)
        print(f"\nSuccess! Extracted data shapes:")
        print(f"  Spikes: {spikes.shape}")
        print(f"  Behavior: {behavior.shape}")
        print(f"  Condition IDs: {condition_ids.shape}")
        print(f"  Number of trials: {len(trial_data)}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
else:
    print("No NWB files found")