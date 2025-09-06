import os
import numpy as np
import scipy.io
from tests.definitions import DIR
from .get_trial_data import get_trial_data
from brn.mint.process_kinematics import process_kinematics
from brn.mint.maximum_likelihood import maximum_likelihood
from brn.mint.fit_poisson_interp import fit_poisson_interp
from brn.mint.recursion import recursion
from brn.mint.estimate_states import estimate_states
from brn.mint.get_state_indices import get_state_indices
from brn.mint.get_time_indices import get_time_indices
from brn.mint.predict import predict



def test(model: dict) -> dict:
    # Load training data
    S, Z, condition = get_trial_data(model["Settings"], "val")

    # Process kinematics.
    Z, unused_variable = process_kinematics(Z, model["Settings"])

    # Determine which spiking observations outside of the test alignment period
    # will be needed to ensure to an estimate can be generated for each sample
    # within the test alignment period.
    test_buffer = np.array((-model["HyperParams"]["obsWindow"] + 1, 0))
    if not model["HyperParams"]["causal"]:
        test_buffer = test_buffer + np.round(
            (model["HyperParams"]["obsWindow"] + model["HyperParams"]["Delta"]) / 2
        )

    # Proceed differently depending on how trials are aligned. For cycling,
    # trials are aligned to two points (move onset & move onset) and vary in
    # length trial-to-trial. Other tasks have trials of equal lengths, aligned
    # just to single event time.

    # HERE WE SKIP THE MC_CYCLE PART
    if model["Settings"]["task"] == "mc_maze":
        # Trim spikes and kinematics to match the expected alignment from the test set plus some buffer.
        buff_align_start = model["Settings"]["testAlignment"][0] + test_buffer[0]
        buff_align_end = model["Settings"]["testAlignment"][-1] + test_buffer[1]
        buffered_alignment = np.arange(buff_align_start, buff_align_end + 1)
        t_mask = np.isin(model["Settings"]["trialAlignment"], buffered_alignment)
        S = S[:, :, t_mask]
        Z = Z[:, :, t_mask]

        # Run MINT
        X_hat, Z_hat = predict(model, S)

        # Remove the buffer so alignment matches desired test alignment
        not_buff_mask = np.isin(buffered_alignment, model["Settings"]["testAlignment"])
        Z = Z[..., not_buff_mask]
        Z_hat = Z_hat[..., not_buff_mask]
        X_hat = X_hat[..., not_buff_mask]

        # Convert neural state estimate to spikes/second
        X_hat /= (model["Delta"] * model["Ts"])

        # Store estimates
        # Note: previously each value was a list of arrays, now just an array.
        estimates = {
            "Z": Z,
            "Z_hat": Z_hat,
            "X_hat": X_hat,
        }
    else:
        estimates = {}

    return estimates

# ===========================================================================
#                                  TESTING
# ===========================================================================
# Unit tests for MINT algorithm components - validates Python implementation
# against MATLAB reference outputs stored in .mat files
# ===========================================================================

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



# Data loading functions (replacing pytest fixtures)
def load_mat_maximum_likelihood_no_restrictedCond_tuple():
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "maximum_likelihood_variables_no_restrictedCond.mat"),
        squeeze_me=True,
    )
    return mat_file

def load_mat_maximum_likelihood_yes_restrictedCond_tuple():
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "maximum_likelihood_variables_yes_restrictedCond.mat"),
        squeeze_me=True,
    )
    return mat_file

def load_mat_fit_poisson_interp_tuple():
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "fit_poisson_interp_variables.mat"), squeeze_me=True
    )
    return mat_file

def load_mat_recursion_less_tuple():
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "recursion_variables_t_prime_less.mat"), squeeze_me=True
    )
    return mat_file

def load_mat_recursion_more_tuple():
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "recursion_variables_t_prime_more.mat"), squeeze_me=True
    )
    return mat_file

def load_mat_estimate_states_tuple():
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "estimate_states_variables.mat"), squeeze_me=True
    )
    return mat_file

def load_mat_get_state_indices_tuple():
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "get_state_indices_variables.mat"), squeeze_me=True
    )
    return mat_file

def load_mat_get_time_indices_tuple():
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "get_time_indices_variables.mat"), squeeze_me=True
    )
    return mat_file

def load_mat_predict_tuple():
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "predict_variables.mat"), squeeze_me=True
    )
    return mat_file

def get_f(load_mat_get_time_indices_tuple):
    mat_file_time = load_mat_get_time_indices_tuple

    t_prime_mat_time = mat_file_time["t_prime"] - 1  # -1 for Python indexing
    T_prime_mat_time = mat_file_time["T_prime"]
    T_mat_time = mat_file_time["T"]
    Delta_mat_time = mat_file_time["Delta"]
    tau_prime_mat_time = mat_file_time["tau_prime"] - 1  # -1 for Python indexing
    causal_mat_time = mat_file_time["causal"]

    t_idx_mat_time_mat = mat_file_time["tIdx"]
    f_mat_time = mat_file_time["f"]

    t_idx_py_time, f_py_time = get_time_indices(
        t_prime_mat_time,
        T_prime_mat_time,
        T_mat_time,
        Delta_mat_time,
        tau_prime_mat_time,
        causal_mat_time,
    )

    return f_py_time

# Test functions
def test_maximum_likelihood_no_restrictedConds():
    mat_file = load_mat_maximum_likelihood_no_restrictedCond_tuple()

    Q_mat = mat_file["Q_in"]
    tau_prime_mat = mat_file["tau_prime"] - 1
    first_idx_mat = mat_file["firstIdx"] - 1
    first_tau_prime_idx_mat = mat_file["firstTauPrimeIdx"] - 1
    restrcited_conds_mat = mat_file["restrictedConds"]  # No restricted Condition

    c_hat_mat = mat_file["c_hat"]
    k_prime_hats_mat = mat_file["k_prime_hats"]

    c_hat_py, k_prime_hats_py = maximum_likelihood(
        Q_mat,
        tau_prime_mat,
        first_idx_mat,
        first_tau_prime_idx_mat,
        restrcited_conds_mat,
    )

    assert (c_hat_py == c_hat_mat - 1) and np.any(
        k_prime_hats_py == k_prime_hats_mat - 1
    ), "Maximum likelihood (no restricted conditions) results do not match"

def test_maximum_likelihood_yes_restrictedConds():
    mat_file = load_mat_maximum_likelihood_yes_restrictedCond_tuple()

    Q_mat = mat_file["Q_in"]
    tau_prime_mat = mat_file["tau_prime"] - 1
    first_idx_mat = mat_file["firstIdx"] - 1
    first_tau_prime_idx_mat = mat_file["firstTauPrimeIdx"] - 1
    restrcited_conds_mat = [mat_file["restrictedConds"] - 1]  # Yes restricted Condition

    c_hat_mat = mat_file["c_hat"]
    k_prime_hats_mat = mat_file["k_prime_hats"]

    c_hat_py, k_prime_hats_py = maximum_likelihood(
        Q_mat,
        tau_prime_mat,
        first_idx_mat,
        first_tau_prime_idx_mat,
        restrcited_conds_mat,
    )

    assert (c_hat_py == c_hat_mat - 1) & np.any(k_prime_hats_py == k_prime_hats_mat - 1), \
        "Maximum likelihood (with restricted conditions) results do not match"

def test_fit_poisson_interp():
    mat_file = load_mat_fit_poisson_interp_tuple()
    S_mat = mat_file["S"]
    X1_mat = mat_file["X1"]
    X2_mat = mat_file["X2"]
    interp_options_mat = mat_file["InterpOptions"]
    default_alpha_mat = mat_file["defaultAlpha"]

    alpha_mat = mat_file["alpha"]

    alpha_py = fit_poisson_interp(
        S_mat, X1_mat, X2_mat, interp_options_mat, default_alpha_mat
    )

    assert np.allclose(alpha_py, alpha_mat), "Fit Poisson interpolation results do not match"

def test_recursion_less_t_prime():
    mat_file = load_mat_recursion_less_tuple()

    Q_in_mat = mat_file["Q_in"]
    s_new_mat = mat_file["s_new"]
    s_old_mat = mat_file["s_old"]
    t_prime_mat = mat_file["t_prime"] - 1
    L_mat = mat_file["L"]
    V_mat = mat_file["V"]
    first_idx_mat = mat_file["firstIdx"] - 1
    shifted_idx1_mat = mat_file["shiftedIdx1"] - 1
    shifted_idx2_mat = mat_file["shiftedIdx2"] - 1
    N_mat = mat_file["N"]
    tau_prime_mat = mat_file["tau_prime"] - 1

    Q_out_mat = mat_file["Q"]

    Q_py = recursion(
        Q_in_mat,
        s_new_mat,
        s_old_mat,
        t_prime_mat,
        L_mat,
        V_mat,
        first_idx_mat,
        shifted_idx1_mat,
        shifted_idx2_mat,
        N_mat,
        tau_prime_mat,
    )

    assert np.allclose(Q_py, Q_out_mat), "Recursion (less t_prime) results do not match"

def test_recursion_more_t_prime():
    mat_file = load_mat_recursion_more_tuple()

    Q_in_mat = mat_file["Q_in"]
    s_new_mat = mat_file["s_new"]
    s_old_mat = mat_file["s_old"]
    t_prime_mat = mat_file["t_prime"] - 1
    L_mat = mat_file["L"]
    V_mat = mat_file["V"]
    first_idx_mat = mat_file["firstIdx"] - 1
    shifted_idx1_mat = mat_file["shiftedIdx1"] - 1
    shifted_idx2_mat = mat_file["shiftedIdx2"] - 1
    N_mat = mat_file["N"]
    tau_prime_mat = mat_file["tau_prime"] - 1

    Q_out_mat = mat_file["Q"]

    Q_py = recursion(
        Q_in_mat,
        s_new_mat,
        s_old_mat,
        t_prime_mat,
        L_mat,
        V_mat,
        first_idx_mat,
        shifted_idx1_mat,
        shifted_idx2_mat,
        N_mat,
        tau_prime_mat,
    )

    assert np.allclose(Q_py, Q_out_mat), "Recursion (more t_prime) results do not match"

def test_estimate_states():
    mat_file = load_mat_estimate_states_tuple()

    # Prepare the model for Python
    model_mat = mat_file["model"]
    headers = (
        "Ts",
        "Delta",
        "dt",
        "tau_prime",
        "obsWindow",
        "causal",
        "Omega_plus",
        "Phi_plus",
        "interp",
        "lambdaRange",
        "nRates",
        "rates",
        "maxSpikes",
        "minProb",
        "minLambda",
        "L",
        "minRate",
        "V",
        "firstIdx",
        "firstTauPrimeIdx",
        "InterpOptions",
        "Settings",
        "HyperParams",
        "MiscParams",
        "shiftedIdx1",
        "shiftedIdx2",
    )
    model_dict = dict()
    for i in range(len(headers)):
        model_dict[headers[i]] = model_mat[()][i]

    # Correcting indexing for Python from MATLAB
    model_dict["firstIdx"] -= 1
    model_dict["tau_prime"] -= 1
    model_dict["firstTauPrimeIdx"] -= 1
    model_dict["shiftedIdx1"] -= 1
    model_dict["shiftedIdx2"] -= 1

    # Preparing matlab variables
    Q_out_mat = mat_file["Q"]
    Q_in_mat = mat_file["Q_in"]
    S_curr_mat = mat_file["S_curr"]
    f_mat = mat_file["f"]

    X_hat_mat = np.stack(mat_file["X_hat"][()])
    Z_hat_mat = np.stack(mat_file["Z_hat"][()])
    C_hat_mat = np.stack(mat_file["C_hat"][()])
    K_hat_mat = np.stack(mat_file["K_hat"][()])
    Alpha_hat_mat = np.stack(mat_file["Alpha_hat"][()])
    f_py = get_f(load_mat_get_time_indices_tuple())

    X_hat_py, Z_hat_py, C_hat_py, K_hat_py, Alpha_hat_py = estimate_states(
        Q_in_mat, S_curr_mat, f_py, model_dict
    )
    X_hat_py = X_hat_py.T
    Z_hat_py = Z_hat_py.T
    C_hat_py = C_hat_py.T
    K_hat_py = K_hat_py.T
    Alpha_hat_py = Alpha_hat_py.T

    assert np.allclose(X_hat_py, X_hat_mat) and np.allclose(Z_hat_py, Z_hat_mat), \
        "Estimate states results do not match"

def test_get_time_and_state_indices():
    mat_file_time = load_mat_get_time_indices_tuple()

    t_prime_mat_time = mat_file_time["t_prime"] - 1  # -1 for Python indexing
    T_prime_mat_time = mat_file_time["T_prime"]
    T_mat_time = mat_file_time["T"]
    Delta_mat_time = mat_file_time["Delta"]
    tau_prime_mat_time = mat_file_time["tau_prime"] - 1  # -1 for Python indexing
    causal_mat_time = mat_file_time["causal"]

    t_idx_mat_time_mat = mat_file_time["tIdx"]
    f_mat_time = mat_file_time["f"]

    t_idx_py_time, f_py_time = get_time_indices(
        t_prime_mat_time,
        T_prime_mat_time,
        T_mat_time,
        Delta_mat_time,
        tau_prime_mat_time,
        causal_mat_time,
    )

    # ---------get_state_indices
    mat_file_state = load_mat_get_state_indices_tuple()

    k_prime_hats_mat_state = (
        mat_file_state["k_prime_hats"] - 1
    )  # -1 for Python indexing
    f_mat_state = mat_file_state["f"]
    K_mat_state = mat_file_state["K"] - 1  # -1 for Python indexing

    kIdx_mat_state = mat_file_state["kIdx"]
    kIdx_py_state = get_state_indices(k_prime_hats_mat_state, f_py_time, K_mat_state)

    assert np.any(t_idx_py_time == t_idx_mat_time_mat - 1) & np.any(
        kIdx_py_state == kIdx_mat_state - 1
    ), "Time and state indices do not match"  # -1 for Python Indexing

def test_predict():
    mat_file = load_mat_predict_tuple()

    # Prepare the model for Python
    model_mat = mat_file["model"]
    headers = (
        "Ts",
        "Delta",
        "dt",
        "tau_prime",
        "obsWindow",
        "causal",
        "Omega_plus",
        "Phi_plus",
        "interp",
        "lambdaRange",
        "nRates",
        "rates",
        "maxSpikes",
        "minProb",
        "minLambda",
        "L",
        "minRate",
        "V",
        "firstIdx",
        "firstTauPrimeIdx",
        "InterpOptions",
        "Settings",
        "HyperParams",
        "MiscParams",
        "shiftedIdx1",
        "shiftedIdx2",
    )
    model_dict = dict()
    for i in range(len(headers)):
        model_dict[headers[i]] = model_mat[()][i]

    # Correcting indexing for Python from MATLAB
    model_dict["firstIdx"] -= 1
    model_dict["tau_prime"] -= 1
    model_dict["firstTauPrimeIdx"] -= 1
    model_dict["shiftedIdx1"] -= 1
    model_dict["shiftedIdx2"] -= 1

    # Preparing matlab variables
    S_mat = np.stack(mat_file["S"][()])

    X_hat_mat = np.stack(mat_file["X_hat"][()])
    Z_hat_mat = np.stack(mat_file["Z_hat"][()])

    X_hat_py, Z_hat_py = predict(model_dict, S_mat)

    assert np.allclose(Z_hat_py, Z_hat_mat, equal_nan=True) and np.allclose(
        X_hat_py, X_hat_mat, equal_nan=True
    ), "Predict results do not match"

def main():
    
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}RUNNING MINT ALGORITHM TESTS{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")
    
    # List of all tests to run
    tests = [
        ("Maximum Likelihood (No Restricted Conditions)", test_maximum_likelihood_no_restrictedConds),
        ("Maximum Likelihood (With Restricted Conditions)", test_maximum_likelihood_yes_restrictedConds),
        ("Fit Poisson Interpolation", test_fit_poisson_interp),
        ("Recursion (Less t_prime)", test_recursion_less_t_prime),
        ("Recursion (More t_prime)", test_recursion_more_t_prime),
        ("Estimate States", test_estimate_states),
        ("Get Time and State Indices", test_get_time_and_state_indices),
        ("Predict", test_predict),
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