from brn.mint.generic_config import generic_config
import os
import sys
import numpy as np
import scipy.io
from tests.definitions import DIR

def mc_maze_config(causal):
    # Get some generic settings and hyperparameters.
    [settings, hyper_params] = generic_config()

    # Determine which time period of raw data should be loaded on each trial,
    # relative to move onset. This should be broad enough to meet all the
    # training needs and provide necessary spiking history for testing.
    settings["trialAlignment"] = np.arange(-800, 901, 1)

    # Determine which time period should be used to evaluate performance,
    # relative to move onset.
    settings["testAlignment"] = np.arange(-250, 451, 1)

    # Hyperparameters related to learning neural trajectories.

    # Set where idealized neural trajectories will begin and end, relative to
    # move onset.
    hyper_params["trajectoriesAlignment"] = np.arange(-500, 701, 1)

    # When learning PSTHs, 'sigma' is the standard deviation of the Gaussian filter.
    hyper_params["sigma"] = 30

    # After learning PSTHs, one may wish to try to further improve the PSTHs.
    # Depending on the data set, it may be beneficial to reduce the
    # dimensionality of the PSTHs across neurons, across conditions, or across
    # trials within each condition. A value of NaN means don't reduce
    # dimensionality.
    hyper_params["nNeuralDims"] = np.nan
    hyper_params["nCondDims"] = 21
    hyper_params["nTrialDims"] = 1  # I typically only ever use NaN or 1

    # Hyperparameters related to inference.

    # What bin size to run inference at. Real-time estimates can still be
    # generated at a higher resolution than this - this value just determines
    # how frequently you update based on new spiking observations.
    hyper_params["Delta"] = 20
    if causal:
        hyper_params["obsWindow"] = 300
    else:
        hyper_params["obsWindow"] = 580

    # The 'causal' hyperparameter is a logical determining whether inference is
    # performed based on a trailing history of spikes (causal = true) or
    # inference is performed based on a spiking window centered around the
    # current time (causal = false).
    hyper_params["causal"] = causal

    return [settings, hyper_params]

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


def main():
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}RUNNING MC_MAZE_CONFIG TESTS{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")
    
    # Run the test
    run_test("MC Maze Config", test_mc_maze_config)
    
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