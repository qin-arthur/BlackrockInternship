"""Diagnose MINT decoder failure with systematic data checks."""

import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from gc_preprocessing import process_nwb_directory
from condition_balancing import balance_conditions, validate_condition_balance
from brn.mint.decoder import MINTDecoder


def diagnose_data_quality(spikes, behavior, condition_ids, verbose=True):
    """Comprehensive data quality diagnostics."""
    
    print("=== DATA QUALITY DIAGNOSTICS ===")
    
    # 1. Basic shape and type checks
    print(f"Data shapes:")
    print(f"  Spikes: {spikes.shape} (should be: trials, neurons, time)")
    print(f"  Behavior: {behavior.shape} (should be: trials, features, time)")
    print(f"  Conditions: {condition_ids.shape} (should be: trials,)")
    print(f"  Data types: spikes={spikes.dtype}, behavior={behavior.dtype}, conditions={condition_ids.dtype}")
    
    # 2. Check for NaNs and infinite values
    spike_nans = np.isnan(spikes).sum()
    behavior_nans = np.isnan(behavior).sum()
    spike_infs = np.isinf(spikes).sum()
    behavior_infs = np.isinf(behavior).sum()
    
    print(f"\nNaN/Inf checks:")
    print(f"  Spike NaNs: {spike_nans} / {spikes.size} ({spike_nans/spikes.size*100:.2f}%)")
    print(f"  Behavior NaNs: {behavior_nans} / {behavior.size} ({behavior_nans/behavior.size*100:.2f}%)")
    print(f"  Spike Infs: {spike_infs}")
    print(f"  Behavior Infs: {behavior_infs}")
    
    # 3. Check data ranges and statistics
    print(f"\nData statistics:")
    print(f"  Spike rates: min={np.nanmin(spikes):.3f}, max={np.nanmax(spikes):.3f}, mean={np.nanmean(spikes):.3f}")
    print(f"  Behavior: min={np.nanmin(behavior):.3f}, max={np.nanmax(behavior):.3f}, mean={np.nanmean(behavior):.3f}")
    print(f"  Spike variance across time: {np.nanvar(spikes, axis=2).mean():.6f}")
    print(f"  Behavior variance across time: {np.nanvar(behavior, axis=2).mean():.6f}")
    
    # 4. Check for all-zero trials or neurons
    zero_spike_trials = np.sum(np.all(spikes == 0, axis=(1,2)))
    zero_behavior_trials = np.sum(np.all(behavior == 0, axis=(1,2)))
    zero_neurons = np.sum(np.all(spikes == 0, axis=(0,2)))
    
    print(f"\nZero data checks:")
    print(f"  All-zero spike trials: {zero_spike_trials} / {spikes.shape[0]}")
    print(f"  All-zero behavior trials: {zero_behavior_trials} / {behavior.shape[0]}")
    print(f"  All-zero neurons: {zero_neurons} / {spikes.shape[1]}")
    
    # 5. Check condition distribution
    unique_conditions, counts = np.unique(condition_ids, return_counts=True)
    print(f"\nCondition distribution:")
    print(f"  Unique conditions: {len(unique_conditions)}")
    print(f"  Trials per condition: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")
    print(f"  Conditions with <5 trials: {np.sum(counts < 5)}")
    
    # 6. Check temporal consistency
    print(f"\nTemporal checks:")
    print(f"  Time bins: {spikes.shape[2]}")
    print(f"  Spike rates over time - first trial: min={spikes[0].min():.3f}, max={spikes[0].max():.3f}")
    print(f"  Behavior over time - first trial: min={behavior[0].min():.3f}, max={behavior[0].max():.3f}")
    
    # 7. Check for constant data
    constant_spike_neurons = np.sum(np.var(spikes, axis=(0,2)) < 1e-10)
    constant_behavior_dims = np.sum(np.var(behavior, axis=(0,2)) < 1e-10)
    
    print(f"\nConstant data checks:")
    print(f"  Constant spike neurons: {constant_spike_neurons} / {spikes.shape[1]}")
    print(f"  Constant behavior dimensions: {constant_behavior_dims} / {behavior.shape[1]}")
    
    # 8. MINT-specific checks
    print(f"\nMINT compatibility checks:")
    
    # Check if spike data looks like rates (should be positive)
    negative_spikes = np.sum(spikes < 0)
    print(f"  Negative spike values: {negative_spikes} (should be 0 for rates)")
    
    # Check typical spike rate ranges (should be reasonable Hz values)
    if np.nanmax(spikes) > 1000:
        print(f"  WARNING: Very high spike rates (max={np.nanmax(spikes):.1f})")
    if np.nanmean(spikes) < 0.01:
        print(f"  WARNING: Very low average spike rates ({np.nanmean(spikes):.6f})")
    
    # Check behavior data (position/velocity should be reasonable)
    print(f"  Position range: x=[{np.nanmin(behavior[:,0]):.3f}, {np.nanmax(behavior[:,0]):.3f}], y=[{np.nanmin(behavior[:,1]):.3f}, {np.nanmax(behavior[:,1]):.3f}]")
    print(f"  Velocity range: x=[{np.nanmin(behavior[:,2]):.3f}, {np.nanmax(behavior[:,2]):.3f}], y=[{np.nanmin(behavior[:,3]):.3f}, {np.nanmax(behavior[:,3]):.3f}]")
    
    return {
        'has_nans': spike_nans > 0 or behavior_nans > 0,
        'has_infs': spike_infs > 0 or behavior_infs > 0,
        'has_zero_trials': zero_spike_trials > 0 or zero_behavior_trials > 0,
        'has_zero_neurons': zero_neurons > 0,
        'has_negative_spikes': negative_spikes > 0,
        'has_constant_data': constant_spike_neurons > 0 or constant_behavior_dims > 0,
        'condition_issues': np.sum(counts < 5) > 0
    }


def test_mint_decoder_step_by_step(spikes, behavior, condition_ids):
    """Test MINT decoder components step by step."""
    
    print("\n=== MINT DECODER STEP-BY-STEP TEST ===")
    
    # Use small subset for testing
    n_test = min(100, spikes.shape[0])
    test_spikes = spikes[:n_test]
    test_behavior = behavior[:n_test] 
    test_conditions = condition_ids[:n_test]
    
    print(f"Testing with {n_test} trials")
    
    # 1. Test template creation
    print(f"\n1. Testing template creation...")
    unique_conditions = np.unique(test_conditions)
    print(f"   Unique conditions in test set: {len(unique_conditions)}")
    
    rate_templates = {}
    behavior_templates = {}
    
    for cond in unique_conditions:
        cond_mask = test_conditions == cond
        n_trials = np.sum(cond_mask)
        if n_trials == 0:
            continue
            
        rate_templates[int(cond)] = np.mean(test_spikes[cond_mask], axis=0)
        behavior_templates[int(cond)] = np.mean(test_behavior[cond_mask], axis=0)
        
        # Check template quality
        rate_template = rate_templates[int(cond)]
        behavior_template = behavior_templates[int(cond)]
        
        print(f"   Condition {cond}: {n_trials} trials")
        print(f"     Rate template: shape={rate_template.shape}, range=[{rate_template.min():.3f}, {rate_template.max():.3f}]")
        print(f"     Behavior template: shape={behavior_template.shape}, range=[{behavior_template.min():.3f}, {behavior_template.max():.3f}]")
        
        # Check for problematic templates
        if np.all(rate_template == 0):
            print(f"     WARNING: All-zero rate template for condition {cond}")
        if np.isnan(rate_template).any():
            print(f"     WARNING: NaN in rate template for condition {cond}")
    
    # 2. Test MINT decoder initialization
    print(f"\n2. Testing MINT decoder initialization...")
    
    from brn.mint.decoder import MINTSettings
    
    try:
        settings = MINTSettings(
            task="diagnostic_test",
            data_path=".",
            results_path=".",
            bin_size=20,
            sampling_period=0.02,
            trial_alignment=np.arange(0, test_spikes.shape[2] * 20, 20),
            trajectories_alignment=np.arange(0, test_spikes.shape[2] * 20, 20),
            test_alignment=np.arange(0, test_spikes.shape[2] * 20, 20),
            observation_window=200,
            causal=True,
            gaussian_sigma=30,
            neural_dims=min(10, test_spikes.shape[1] // 2),
            condition_dims=min(5, len(unique_conditions)),
            trial_dims=1,
            min_lambda=1e-6,
        )
        print(f"   MINT settings created successfully")
        print(f"   Neural dims: {settings.neural_dims}")
        print(f"   Condition dims: {settings.condition_dims}")
        
        decoder = MINTDecoder(settings)
        print(f"   MINT decoder initialized successfully")
        
    except Exception as e:
        print(f"   ERROR: MINT initialization failed: {e}")
        return False
    
    # 3. Test template mode training
    print(f"\n3. Testing template mode training...")
    
    try:
        decoder.fit(
            rate_templates=rate_templates,
            behavior_templates=behavior_templates,
            condition_list=unique_conditions
        )
        print(f"   Template mode training completed successfully")
        
    except Exception as e:
        print(f"   ERROR: Template mode training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. Test prediction
    print(f"\n4. Testing prediction...")
    
    try:
        X_hat, Z_hat = decoder.predict(test_spikes)
        print(f"   Prediction completed successfully")
        print(f"   Predicted neural shape: {X_hat.shape}")
        print(f"   Predicted behavior shape: {Z_hat.shape}")
        
        # Check prediction quality
        neural_range = f"[{X_hat.min():.3f}, {X_hat.max():.3f}]"
        behavior_range = f"[{Z_hat.min():.3f}, {Z_hat.max():.3f}]"
        print(f"   Predicted neural range: {neural_range}")
        print(f"   Predicted behavior range: {behavior_range}")
        
        if np.isnan(X_hat).any() or np.isnan(Z_hat).any():
            print(f"   WARNING: NaN values in predictions")
            
        return True
        
    except Exception as e:
        print(f"   ERROR: Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run comprehensive MINT failure diagnosis."""
    
    print("MINT DECODER FAILURE DIAGNOSIS")
    print("=" * 50)
    
    # Load data
    nwb_dir = Path("C:/Users/aqin/Downloads/GeneralizedClick/mint_pipeline_output/nwb_files")
    
    print(f"Loading data from {nwb_dir}...")
    sessions = process_nwb_directory(nwb_dir, merge_sessions=True, verbose=False)
    
    if "merged" in sessions:
        session_data = sessions["merged"]
        if len(session_data) == 4:
            spikes, behavior, condition_ids, trial_data = session_data
        else:
            spikes, behavior, condition_ids = session_data
            trial_data = None
    else:
        session_name = list(sessions.keys())[0]
        session_data = sessions[session_name]
        if len(session_data) == 4:
            spikes, behavior, condition_ids, trial_data = session_data
        else:
            spikes, behavior, condition_ids = session_data
            trial_data = None
        print(f"Using single session: {session_name}")
    
    print(f"Loaded {spikes.shape[0]} trials")
    
    # Run diagnostics
    issues = diagnose_data_quality(spikes, behavior, condition_ids)
    
    # Run step-by-step MINT test
    success = test_mint_decoder_step_by_step(spikes, behavior, condition_ids)
    
    # Summary
    print(f"\n=== DIAGNOSIS SUMMARY ===")
    print(f"Data quality issues found:")
    for issue, present in issues.items():
        status = "YES" if present else "NO"
        print(f"  {issue}: {status}")
    
    print(f"\nMINT decoder test: {'PASSED' if success else 'FAILED'}")
    
    if not success:
        print(f"\nNext steps:")
        print(f"1. Fix data quality issues identified above")
        print(f"2. Check NWB data extraction process")
        print(f"3. Verify MINT settings are appropriate for data")


if __name__ == "__main__":
    main()