"""Quick test of the fixed main.py implementation."""

import numpy as np
from main import main

def test_fixed_implementation():
    """Test the fixed implementation on a single file."""
    
    print("=" * 60)
    print("TESTING FIXED IMPLEMENTATION")
    print("=" * 60)
    
    # Test with single file
    nwb_path = "C:/Users/aqin/Downloads/GeneralizedClick/mint_pipeline_output/nwb_files/DATA_P2Lab_1587.nwb"
    output_dir = "C:/Users/aqin/Downloads/GeneralizedClick/mint_pipeline_output/mint_results_final_test"
    
    try:
        decoder, results = main(
            nwb_path=nwb_path,
            output_dir=output_dir,
            merge_sessions=False,
            test_fraction=0.2,
            verbose=True
        )
        
        print(f"\n" + "=" * 40)
        print("SUCCESS! Fixed implementation completed.")
        print(f"Final R² scores:")
        print(f"  Neural R²: {results['neural_r2']:.6f}")
        print(f"  Behavior R²: {results['behavior_r2']:.6f}")
        print(f"  Position R²: {results['position_r2']:.6f}")
        print(f"  Velocity R²: {results['velocity_r2']:.6f}")
        print(f"Results saved to: {output_dir}")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_fixed_implementation()
    if success:
        print("\n✅ All fixes successfully integrated!")
    else:
        print("\n❌ Issues remain - check error output above")