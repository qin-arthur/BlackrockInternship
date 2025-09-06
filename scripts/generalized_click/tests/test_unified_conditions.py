"""Test unified condition mapping across multiple files."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from gc_preprocessing import process_nwb_directory
import numpy as np

# Test on multiple files
nwb_dir = Path("C:/Users/aqin/Downloads/GeneralizedClick/mint_pipeline_output/nwb_files")
nwb_files = list(nwb_dir.glob("*.nwb"))

if len(nwb_files) >= 2:
    print(f"Testing unified conditions on first 2 files from {len(nwb_files)} available files")
    
    # Create a test directory with just 2 files to speed up testing
    test_dir = Path("test_nwb_subset")
    test_dir.mkdir(exist_ok=True)
    
    # Copy just 2 files for testing (using symlinks to save space)
    import shutil
    test_files = []
    for i, nwb_file in enumerate(nwb_files[:2]):
        test_file = test_dir / nwb_file.name
        if not test_file.exists():
            shutil.copy2(nwb_file, test_file)
        test_files.append(test_file)
    
    print(f"Testing unified conditions with files:")
    for f in test_files:
        print(f"  {f.name}")
    
    # Process with unified conditions
    sessions = process_nwb_directory(
        nwb_dir=test_dir,
        merge_sessions=True,
        verbose=True
    )
    
    print(f"\n=== Results ===")
    for session_name, session_data in sessions.items():
        if len(session_data) >= 3:
            spikes, behavior, condition_ids = session_data[:3]
            unique_conditions = np.unique(condition_ids)
            print(f"{session_name}:")
            print(f"  Trials: {len(condition_ids)}")
            print(f"  Unique conditions: {len(unique_conditions)}")
            print(f"  Condition range: {unique_conditions.min()}-{unique_conditions.max()}")
            
            # Show condition distribution
            for cond_id in unique_conditions[:10]:  # Show first 10
                count = np.sum(condition_ids == cond_id)
                print(f"    Condition {cond_id}: {count} trials")
            if len(unique_conditions) > 10:
                print(f"    ... and {len(unique_conditions) - 10} more conditions")
    
    # Cleanup
    shutil.rmtree(test_dir)
    
else:
    print(f"Need at least 2 NWB files for testing, found {len(nwb_files)}")