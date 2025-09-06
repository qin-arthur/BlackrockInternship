import os
import numpy as np
import scipy.io
from tests.definitions import DIR
from brn.mint.mint import MINT
from brn.mint.fit import fit
from .get_trial_data import get_trial_data
from brn.mint.gauss_filt import gauss_filt
from brn.mint.process_kinematics import process_kinematics
from brn.mint.fit_trajectories import fit_trajectories
from brn.mint.smooth_average import smooth_average
from brn.mint.get_rate_indices import get_rate_indices
from brn.mint.bin_data import bin_data
from .mc_maze_config import mc_maze_config

def train(settings, hyper_params):
    # Load training data
    S, Z, condition = get_trial_data(settings, "train")

    # Don't optimize hyperparameters, just use those provided. But any
    # hyperparameter selection method could be used here (e.g., grid search).
    train_summary = {"HyperParams": hyper_params}

    # Train model
    model_a = MINT(settings, train_summary["HyperParams"])  # L calculated
    model = fit(model_a, S, Z, condition)  # Omega_plus, Phi_plus, V calculated

    # Store kinematic labels.
    train_summary["kinematic_labels"] = model["MiscParams"]["kin_labels"]

    return model, train_summary


# ============================================================================
# ================================ TESTING ===================================
# ============================================================================


# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Test result tracking
test_results = []

def run_test(test_name, test_function):
    """Run a single test and track results"""
    print(f"\n{Colors.OKBLUE}Running test: {test_name}{Colors.ENDC}")
    try:
        test_function()
        print(f"{Colors.OKGREEN}✓ PASSED: {test_name}{Colors.ENDC}")
        test_results.append((test_name, True, None))
        return True
    except AssertionError as e:
        print(f"{Colors.FAIL}✗ FAILED: {test_name}{Colors.ENDC}")
        print(f"{Colors.WARNING}  Assertion Error: {str(e)}{Colors.ENDC}")
        test_results.append((test_name, False, f"Assertion Error: {str(e)}"))
        return False
    except Exception as e:
        print(f"{Colors.FAIL}✗ ERROR: {test_name}{Colors.ENDC}")
        print(f"{Colors.WARNING}  Error: {type(e).__name__}: {str(e)}{Colors.ENDC}")
        test_results.append((test_name, False, f"{type(e).__name__}: {str(e)}"))
        return False

def print_summary():
    """Print a summary of all test results"""
    total_tests = len(test_results)
    passed_tests = sum(1 for _, passed, _ in test_results if passed)
    failed_tests = total_tests - passed_tests
    
    print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}TEST SUMMARY{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")
    
    print(f"\nTotal tests run: {total_tests}")
    print(f"{Colors.OKGREEN}Passed: {passed_tests}{Colors.ENDC}")
    print(f"{Colors.FAIL}Failed: {failed_tests}{Colors.ENDC}")
    
    if failed_tests > 0:
        print(f"\n{Colors.FAIL}Failed tests:{Colors.ENDC}")
        for test_name, passed, error in test_results:
            if not passed:
                print(f"  - {test_name}")
                if error:
                    print(f"    {Colors.WARNING}{error}{Colors.ENDC}")
    
    print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
    
    return passed_tests == total_tests


# Data loading functions
def config_tuple():
    [settings, hyper_params] = mc_maze_config(True)
    return settings, hyper_params

def load_mat_Post_Training_50_tuple():
    fpath = os.path.join(DIR, "Post_training_50.mat")
    mat_file = scipy.io.loadmat(fpath, squeeze_me=True)
    return mat_file

def load_mat_50_post_gauss_filt_tuple():
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "gauss_filt_variables.mat"), squeeze_me=True
    )
    return mat_file

def load_mat_process_kinematics_tuple():
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "process_kinematics_variables.mat"), squeeze_me=True
    )
    return mat_file

def load_mat_50_post_fit_trajectories_tuple():
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "fit_trajectories_variables.mat"), squeeze_me=True
    )
    return mat_file

def load_mat_50_smooth_average_tuple():
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "smooth_average_variables.mat"), squeeze_me=True
    )
    return mat_file

def load_mat_50_get_rate_indices_tuple():
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "get_rate_indices_variables.mat"), squeeze_me=True
    )
    return mat_file

def load_mat_50_bin_data_tuple():
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "binned_data_variables.mat"), squeeze_me=True
    )
    return mat_file

# Test functions
def test_mc_maze_config():
    settings, hyper_params = config_tuple()
    mat_file = load_mat_Post_Training_50_tuple()

    settings_mat = np.array(mat_file["Settings"])
    hyper_params_mat = np.array(mat_file["HyperParams"])

    assert settings["task"] == settings_mat["task"], "Task mismatch"
    assert settings["Ts"] == settings_mat["Ts"], "Ts mismatch"
    assert np.allclose(settings["trialAlignment"], settings_mat["trialAlignment"][()]), "Trial alignment mismatch"
    assert np.allclose(settings["testAlignment"], settings_mat["testAlignment"][()]), "Test alignment mismatch"

    assert np.allclose(
        hyper_params["trajectoriesAlignment"],
        hyper_params_mat["trajectoriesAlignment"][()],
    ), "Trajectories alignment mismatch"
    assert np.isnan(hyper_params_mat["nNeuralDims"][()]) & np.isnan(
        hyper_params["nNeuralDims"]
    ), "nNeuralDims mismatch"

    for hyper in [
        "softNorm",
        "minProb",
        "minLambda",
        "minRate",
        "interp",
        "sigma",
        "nCondDims",
        "nTrialDims",
        "Delta",
        "obsWindow",
        "causal",
    ]:
        assert hyper_params[hyper] == hyper_params_mat[hyper], f"{hyper} mismatch"

def test_gauss_filt():
    mat_file = load_mat_50_post_gauss_filt_tuple()
    binSize_mat = mat_file["binSize"]
    sigma_mat = mat_file["sigma"]
    spikes_mat = mat_file["spikes"]
    filtSpikes_mat = mat_file["filtSpikes"]

    filtSpikes_py = gauss_filt(spikes_mat, sigma_mat, binSize_mat)

    assert np.allclose(filtSpikes_py, filtSpikes_mat), "Gauss filter results do not match"

def test_process_kinematics():
    mat_file = load_mat_process_kinematics_tuple()
    Z_mat = np.stack(mat_file["Z"])
    settings_mat = mat_file["Settings"]
    Z_out_mat = np.stack(mat_file["Z_out"])
    labels_mat = mat_file["labels"]
    Z_py, labels_py = process_kinematics(Z_mat, settings_mat[()])

    assert labels_py == labels_mat.tolist(), "Kinematic labels do not match"
    assert np.allclose(Z_py, Z_out_mat), "Processed kinematics do not match"

def test_fit_trajectories():
    mat_file = load_mat_50_post_fit_trajectories_tuple()
    S_mat = np.stack(mat_file["S"])
    Z_mat = np.stack(mat_file["Z"])
    condition_mat = mat_file["condition"]
    hyper_params_mat = mat_file["HyperParams"][()]
    settings_mat = mat_file["Settings"][()]

    Omega_plus_mat = np.stack(mat_file["Omega_plus"])
    Phi_plus_mat = np.stack(mat_file["Phi_plus"])
    MiscParams_mat = mat_file["MiscParams"][()][0]

    Omega_plus, Phi_plus, MiscParams = fit_trajectories(
        S_mat, Z_mat, condition_mat, settings_mat, hyper_params_mat
    )

    assert np.allclose(Omega_plus_mat, np.stack([_ for _ in Omega_plus.values()])), "Omega_plus does not match"
    assert np.allclose(Phi_plus_mat, np.stack([_ for _ in Phi_plus.values()])), "Phi_plus does not match"
    assert MiscParams_mat.tolist() == MiscParams["kin_labels"], "MiscParams labels do not match"

def test_smooth_average():
    mat_file = load_mat_50_smooth_average_tuple()
    hyper_params_mat = mat_file["HyperParams"][()]
    X_dict = {ix + 1: np.stack(v) if v.dtype == "O" else v[None, ...] for ix, v in enumerate(mat_file["X_in"])}
    Ts_mat = mat_file["Ts"]
    X_bar_mat = np.stack(mat_file["X_bar"])

    X_bar = smooth_average(X_dict, hyper_params_mat, Ts_mat)

    assert np.allclose(X_bar_mat, np.stack([v for v in X_bar.values()])), "Smooth average results do not match"

def test_get_rate_indices():
    mat_file = load_mat_50_get_rate_indices_tuple()
    lambda_in_mat = mat_file["lambda_in"]
    lambdaRange_mat = mat_file["lambdaRange"]
    nRates_mat = mat_file["nRates"]

    v = get_rate_indices(lambda_in_mat, lambdaRange_mat, nRates_mat)
    v_mat = mat_file["v"]

    assert np.allclose(v_mat, v), "Rate indices do not match"

def test_bin_data():
    mat_file = load_mat_50_bin_data_tuple()
    data_mat = mat_file["data"]
    binSize_mat = mat_file["binSize"]
    method_mat = mat_file["method"]

    binnedData = bin_data(data_mat, binSize_mat, method_mat)
    binnedData_mat = mat_file["binnedData"]

    assert np.allclose(binnedData_mat, binnedData), "Binned data does not match"


def main():
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}RUNNING TRAIN.PY TESTS{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")
    
    # List of all tests to run
    tests = [
        ("MC Maze Config", test_mc_maze_config),
        ("Gauss Filter", test_gauss_filt),
        ("Process Kinematics", test_process_kinematics),
        ("Fit Trajectories", test_fit_trajectories),
        ("Smooth Average", test_smooth_average),
        ("Get Rate Indices", test_get_rate_indices),
        ("Bin Data", test_bin_data),
    ]
    
    # Run all tests
    for test_name, test_func in tests:
        run_test(test_name, test_func)
    
    # Print summary
    all_passed = print_summary()
    
    # Exit with appropriate code
    if all_passed:
        print(f"\n{Colors.OKGREEN}All tests passed!{Colors.ENDC}")
        exit(0)
    else:
        print(f"\n{Colors.FAIL}Some tests failed!{Colors.ENDC}")
        exit(1)

if __name__ == "__main__":
    main()