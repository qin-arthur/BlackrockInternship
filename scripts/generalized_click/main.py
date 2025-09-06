import numpy as np
from pathlib import Path
import argparse
from typing import Union
from brn.mint.model import MINT, MINTSettings
from gc_preprocess_new import process_nwb_directory, process_single_nwb, split_train_test


def main(nwb_path: Union[str, Path], 
         output_dir: str,
         merge_sessions: bool = True,
         test_fraction: float = 0.2,
         verbose: bool = True):
    """Train MINT decoder on generalized click NWB data.
    
    Args:
        nwb_path: Path to NWB file or directory containing NWB files
        output_dir: Directory to save results
        merge_sessions: Whether to merge multiple sessions
        test_fraction: Fraction of data to use for testing
        verbose: Whether to print progress
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("=" * 50)
        print("MINT Decoder Training Pipeline")
        print("=" * 50)
    
    # Process NWB data
    nwb_path = Path(nwb_path)
    
    if nwb_path.is_file():
        # Single NWB file
        if verbose:
            print(f"Processing single NWB file: {nwb_path}")
        spikes, behavior, cond_ids, trial_data = process_single_nwb(nwb_path, verbose=verbose)
        
    elif nwb_path.is_dir():
        # Directory of NWB files
        if verbose:
            print(f"Processing NWB directory: {nwb_path}")
        
        sessions = process_nwb_directory(
            nwb_dir=nwb_path,
            merge_sessions=merge_sessions,
            verbose=verbose
        )
        
        # Use merged data if available, otherwise first session
        if "merged" in sessions:
            session_data = sessions["merged"]
            if len(session_data) == 4:
                spikes, behavior, cond_ids, trial_data = session_data
            else:
                spikes, behavior, cond_ids = session_data
                trial_data = []  # No trial data for merged sessions
        else:
            session_name = list(sessions.keys())[0]
            session_data = sessions[session_name]
            if len(session_data) == 4:
                spikes, behavior, cond_ids, trial_data = session_data
            else:
                spikes, behavior, cond_ids = session_data
                trial_data = []
            if verbose:
                print(f"Using data from session: {session_name}")
    else:
        raise ValueError(f"Path does not exist: {nwb_path}")
    
    if verbose:
        print(f"\nLoaded data:")
        print(f"  Spikes: {spikes.shape}")
        print(f"  Behavior: {behavior.shape}")
        print(f"  Conditions: {len(np.unique(cond_ids))} unique conditions")
    
    
    # Split into train/test
    train_spikes, train_behavior, train_conds, test_spikes, test_behavior, test_conds = split_train_test(
        spikes=spikes,
        behavior=behavior,
        condition_ids=cond_ids,
        test_fraction=test_fraction,
        verbose=verbose
    )
    
    # Create MINT settings
    # if trial_data:
    #     settings = create_mint_settings(
    #         train_conds=train_conds,
    #         train_spikes=train_spikes,
    #         train_behavior=train_behavior,
    #         trial_data=trial_data,
    #         output_path=output_path
    #     )
    # else:
    #     # Fallback for merged data without trial_data
    #     print("    Warning: No trial data available for dynamic settings, using defaults")
    #     from brn.mint.decoder import MINTSettings
    #     settings = MINTSettings(
    #         task="generalized_click",
    #         data_path=str(output_path / "data"),
    #         results_path=str(output_path / "results"),
    #         bin_size=20,
    #         sampling_period=0.02,
    #         trial_alignment=np.arange(0, train_spikes.shape[2] * 20, 20),
    #         trajectories_alignment=np.arange(0, train_spikes.shape[2] * 20, 20),
    #         test_alignment=np.arange(0, train_spikes.shape[2] * 20, 20),
    #         observation_window=200,
    #         causal=True,
    #         gaussian_sigma=30,
    #         neural_dims=min(15, train_spikes.shape[1] // 2),
    #         condition_dims=min(8, len(np.unique(train_conds)) * 2),
    #         trial_dims=min(3, np.min([np.sum(train_conds == c) for c in np.unique(train_conds)]) - 1),
    #         min_lambda=1e-8,
    #     )
    
    # if verbose:
    #     print(f"\nMINT Settings:")
    #     print(f"  Observation window: {settings.observation_window}ms")
    #     print(f"  Causal: {settings.causal}")
    #     print(f"  Neural dims: {settings.neural_dims}")
    #     print(f"  Condition dims: {settings.condition_dims}")
    #     print(f"  Trial dims: {settings.trial_dims}")
    
    
    settings = MINTSettings(
        task="mc_maze",
        fs=50.0,
        obs_window=500,
        min_lambda=1.0,
        n_rates=2000,
        min_rate=0.0,
        max_rate=20,
        min_prob=1e-6,
        interp_mode=3,
        interp_max_iters=10,
        interp_tolerance=0.01
    )
    
    # Initialize decoder
    decoder = MINT(settings)
    
    decoder.fit(
        train_spikes,
        train_behavior,
        train_conds
    )
    
    if verbose:
        print("Training complete!")
        
    for i in test_spikes:
        for j in test_spikes[i]:
            decoder.predict(test_spikes[i][j])
    
    
    return
    # # Test decoder
    # if verbose:
    #     print(f"\nTesting decoder on {len(test_spikes)} held-out trials...")
        
    # # Predict on test data
    # X_hat, Z_hat = decoder.predict(test_spikes)
    
    # # Calculate metrics
    # neural_r2 = calculate_r2(test_spikes, X_hat)
    # behavior_r2 = calculate_r2(test_behavior, Z_hat)
    # position_r2 = calculate_r2(test_behavior[:, :2], Z_hat[:, :2])
    # velocity_r2 = calculate_r2(test_behavior[:, 2:], Z_hat[:, 2:])
    
    # if verbose:
    #     print(f"\nTest Results:")
    #     print(f"  Neural R²: {neural_r2:.3f}")
    #     print(f"  Behavior R²: {behavior_r2:.3f}")
    #     print(f"  Position R²: {position_r2:.3f}")
    #     print(f"  Velocity R²: {velocity_r2:.3f}")
    
    # # Save results with comprehensive diagnostics
    # results = {
    #     'settings': settings.__dict__,
    #     'neural_r2': neural_r2,
    #     'behavior_r2': behavior_r2,
    #     'position_r2': position_r2,
    #     'velocity_r2': velocity_r2,
    #     'train_conditions': train_conds,
    #     'test_conditions': test_conds,
    #     'test_predictions': {
    #         'neural': X_hat,
    #         'behavior': Z_hat,
    #         'true_behavior': test_behavior,
    #         'true_neural': test_spikes,
    #     },
    #     'diagnostics': {
    #         'condition_validation': condition_validation,
    #         'balancing_stats': balancing_stats,
    #         'original_data_shapes': {
    #             'spikes': spikes.shape,
    #             'behavior': behavior.shape,
    #             'conditions': len(np.unique(cond_ids))
    #         },
    #         'final_data_shapes': {
    #             'train_spikes': train_spikes.shape,
    #             'train_behavior': train_behavior.shape,
    #             'test_spikes': test_spikes.shape,
    #             'test_behavior': test_behavior.shape
    #         }
    #     }
    # }
    
    # # Add movement alignment diagnostics if available
    # if hasattr(settings, 'movement_stats'):
    #     results['diagnostics']['movement_stats'] = settings.movement_stats
    
    # np.savez(
    #     output_path / "results.npz",
    #     **results
    # )
    
    # if verbose:
    #     print(f"\nResults saved to {output_path / 'results.npz'}")
    #     print("=" * 50)
    
    # return decoder, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MINT decoder on generalized click NWB data")
    parser.add_argument("nwb_path", type=str, nargs='?',
                      default="C:/Users/aqin/Downloads/GeneralizedClick/mint_pipeline_output/nwb_files",
                      help="Path to NWB file or directory containing NWB files (default: C:/Users/aqin/Downloads/GeneralizedClick/mint_pipeline_output/nwb_files)")
    parser.add_argument("output_dir", type=str, nargs='?',
                      default="C:/Users/aqin/Downloads/GeneralizedClick/mint_pipeline_output/mint_results",
                      help="Output directory for results (default: C:/Users/aqin/Downloads/GeneralizedClick/mint_pipeline_output/mint_results)")
    parser.add_argument("--no-merge", action="store_true",
                      help="Don't merge multiple sessions (use first session only)")
    parser.add_argument("--test-fraction", type=float, default=0.2,
                      help="Fraction of data for testing (default: 0.2)")
    parser.add_argument("--quiet", action="store_true",
                      help="Suppress output")
    
    args = parser.parse_args()
    
    # Run training
    main(
        nwb_path=args.nwb_path,
        output_dir=args.output_dir,
        merge_sessions=not args.no_merge,
        test_fraction=args.test_fraction,
        verbose=not args.quiet
    )