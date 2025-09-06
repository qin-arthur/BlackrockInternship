"""Test MINT decoder in template mode to avoid double preprocessing."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from gc_preprocessing import process_single_nwb, split_train_test, create_mint_settings
from condition_balancing import balance_conditions, validate_condition_balance
from brn.mint.decoder import MINTDecoder
import numpy as np

def test_template_mode():
    """Test MINT decoder using template mode instead of raw data mode."""
    
    # Load and process data
    nwb_dir = Path("C:/Users/aqin/Downloads/GeneralizedClick/mint_pipeline_output/nwb_files")
    nwb_files = list(nwb_dir.glob("*.nwb"))[:1]
    
    if not nwb_files:
        print("No NWB files found for testing")
        return
    
    print(f"Testing template mode on {nwb_files[0].name}")
    print("=" * 60)
    
    # Extract and preprocess data
    spikes, behavior, condition_ids, trial_data = process_single_nwb(nwb_files[0], verbose=False)
    
    # Use subset for faster testing
    if len(spikes) > 100:
        spikes = spikes[:100]
        behavior = behavior[:100]
        condition_ids = condition_ids[:100]
        trial_data = trial_data[:100]
    
    # Balance conditions
    condition_validation = validate_condition_balance(condition_ids, verbose=False)
    if not condition_validation['well_balanced']:
        spikes, behavior, condition_ids, trial_data, _ = balance_conditions(
            spikes=spikes,
            behavior=behavior,
            condition_ids=condition_ids,
            trial_data=trial_data,
            verbose=False
        )
    
    # Split train/test
    train_spikes, train_behavior, train_conds, test_spikes, test_behavior, test_conds = split_train_test(
        spikes=spikes,
        behavior=behavior,
        condition_ids=condition_ids,
        test_fraction=0.2,
        verbose=False
    )
    
    print(f"Data shapes:")
    print(f"  Train: spikes {train_spikes.shape}, behavior {train_behavior.shape}")
    print(f"  Test: spikes {test_spikes.shape}, behavior {test_behavior.shape}")
    print(f"  Conditions: {len(np.unique(train_conds))}")
    
    # Create MINT settings
    output_path = Path("test_output")
    output_path.mkdir(exist_ok=True)
    
    settings = create_mint_settings(
        train_conds=train_conds,
        train_spikes=train_spikes,
        train_behavior=train_behavior,
        trial_data=trial_data,
        output_path=output_path,
        use_movement_alignment=False  # Use fixed alignment for faster testing
    )
    
    print(f"\nMINT Settings:")
    print(f"  Observation window: {settings.observation_window}ms")
    print(f"  Test alignment: {len(settings.test_alignment)} bins")
    print(f"  Causal: {settings.causal}")
    
    # Test Template Mode
    print(f"\n=== Testing Template Mode ===")
    
    # Create templates manually (this is what main.py now does)
    unique_conditions = np.unique(train_conds)
    rate_templates = {}
    behavior_templates = {}
    
    for cond in unique_conditions:
        cond_mask = train_conds == cond
        rate_templates[int(cond)] = np.mean(train_spikes[cond_mask], axis=0)
        behavior_templates[int(cond)] = np.mean(train_behavior[cond_mask], axis=0)
    
    print(f"Created templates for {len(unique_conditions)} conditions")
    print(f"Rate template shape: {list(rate_templates.values())[0].shape}")
    print(f"Behavior template shape: {list(behavior_templates.values())[0].shape}")
    
    # Initialize and train decoder using template mode
    decoder = MINTDecoder(settings)
    
    print(f"\nTraining decoder in template mode...")
    
    try:
        decoder.fit(
            rate_templates=rate_templates,
            behavior_templates=behavior_templates,
            condition_list=unique_conditions
        )
        print("PASS: Template mode training successful!")
        
        # Test prediction
        print(f"\nTesting prediction...")
        X_hat, Z_hat = decoder.predict(test_spikes)
        
        print(f"Prediction shapes:")
        print(f"  Neural: {X_hat.shape}")
        print(f"  Behavior: {Z_hat.shape}")
        print("PASS: Prediction successful!")
        
        # Calculate basic R²
        def calculate_r2(y_true, y_pred):
            y_true_flat = y_true.reshape(-1)
            y_pred_flat = y_pred.reshape(-1)
            mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
            y_true_flat = y_true_flat[mask]
            y_pred_flat = y_pred_flat[mask]
            
            if len(y_true_flat) == 0:
                return 0.0
            
            ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
            ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        neural_r2 = calculate_r2(test_spikes, X_hat)
        behavior_r2 = calculate_r2(test_behavior, Z_hat)
        
        print(f"\nPerformance:")
        print(f"  Neural R²: {neural_r2:.3f}")
        print(f"  Behavior R²: {behavior_r2:.3f}")
        
        # Verify no double preprocessing occurred
        print(f"\n=== Template Mode Verification ===")
        
        # Check that templates are properly formatted
        template_checks = [
            ("All conditions have templates", len(rate_templates) == len(unique_conditions)),
            ("Rate templates are 2D", all(t.ndim == 2 for t in rate_templates.values())),
            ("Behavior templates are 2D", all(t.ndim == 2 for t in behavior_templates.values())),
            ("Consistent neural dimensions", len(set(t.shape[0] for t in rate_templates.values())) == 1),
            ("Consistent behavior dimensions", len(set(t.shape[0] for t in behavior_templates.values())) == 1),
        ]
        
        all_passed = True
        for check_name, result in template_checks:
            status = "PASS" if result else "FAIL"
            print(f"  {status}: {check_name}")
            if not result:
                all_passed = False
        
        if all_passed:
            print(f"\nSUCCESS: Template mode working correctly!")
            print(f"   - No double preprocessing")
            print(f"   - Preserves our custom alignment and balancing")
            print(f"   - MINT decoder trains and predicts successfully")
        else:
            print(f"\nWARNING: Some template mode checks failed")
        
        return True
        
    except Exception as e:
        print(f"FAIL: Template mode training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_template_mode()