import os
import numpy as np
import pytest
import scipy.io
from pathlib import Path

from brn.mint import utils
from tests.definitions import DIR

# ============================================================================
# Pytest fixtures for data loading
# ============================================================================

@pytest.fixture
def gauss_filt_data():
    """Load test data for Gaussian filter"""
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "gauss_filt_variables.mat"), squeeze_me=True
    )
    return mat_file

@pytest.fixture
def process_kinematics_data():
    """Load test data for process kinematics"""
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "process_kinematics_variables.mat"), squeeze_me=True
    )
    return mat_file

@pytest.fixture
def bin_data_data():
    """Load test data for bin_data function"""
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "binned_data_variables.mat"), squeeze_me=True
    )
    return mat_file

@pytest.fixture
def get_rate_indices_data():
    """Load test data for get_rate_indices"""
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "get_rate_indices_variables.mat"), squeeze_me=True
    )
    return mat_file

@pytest.fixture
def smooth_average_data():
    """Load test data for smooth_average"""
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "smooth_average_variables.mat"), squeeze_me=True
    )
    return mat_file

@pytest.fixture
def fit_poisson_interp_data():
    """Load test data for fit_poisson_interp"""
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "fit_poisson_interp_variables.mat"), squeeze_me=True
    )
    return mat_file

@pytest.fixture
def recursion_less_data():
    """Load test data for recursion (less t_prime case)"""
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "recursion_variables_t_prime_less.mat"), squeeze_me=True
    )
    return mat_file

@pytest.fixture
def recursion_more_data():
    """Load test data for recursion (more t_prime case)"""
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "recursion_variables_t_prime_more.mat"), squeeze_me=True
    )
    return mat_file

@pytest.fixture
def get_time_indices_data():
    """Load test data for get_time_indices"""
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "get_time_indices_variables.mat"), squeeze_me=True
    )
    return mat_file

@pytest.fixture
def get_state_indices_data():
    """Load test data for get_state_indices"""
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "get_state_indices_variables.mat"), squeeze_me=True
    )
    return mat_file

@pytest.fixture
def maximum_likelihood_no_restricted_data():
    """Load test data for maximum_likelihood (no restricted conditions)"""
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "maximum_likelihood_variables_no_restrictedCond.mat"),
        squeeze_me=True,
    )
    return mat_file

@pytest.fixture
def maximum_likelihood_yes_restricted_data():
    """Load test data for maximum_likelihood (with restricted conditions)"""
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "maximum_likelihood_variables_yes_restrictedCond.mat"),
        squeeze_me=True,
    )
    return mat_file

# ============================================================================
# Core MINT utils functionality tests
# ============================================================================

def test_gauss_filt(gauss_filt_data):
    """Test Gaussian filtering function"""
    mat_file = gauss_filt_data
    binSize_mat = mat_file["binSize"]
    sigma_mat = mat_file["sigma"]
    spikes_mat = mat_file["spikes"]
    filtSpikes_mat = mat_file["filtSpikes"]

    filtSpikes_py = utils.gauss_filt(spikes_mat, sigma_mat, binSize_mat)

    assert np.allclose(filtSpikes_py, filtSpikes_mat)

def test_process_kinematics(process_kinematics_data):
    """Test kinematic processing function"""
    mat_file = process_kinematics_data
    Z_mat = np.stack(mat_file["Z"])
    settings_mat = mat_file["Settings"][()]
    Z_out_mat = np.stack(mat_file["Z_out"])
    
    # Extract trial alignment from settings
    trial_alignment = settings_mat["trialAlignment"]
    
    Z_py = utils.process_kinematics(Z_mat, trial_alignment)

    # Only test the behavior output now, not labels
    assert np.allclose(Z_py, Z_out_mat)

def test_bin_data(bin_data_data):
    """Test data binning function"""
    mat_file = bin_data_data
    data_mat = mat_file["data"]
    binSize_mat = mat_file["binSize"]
    method_mat = mat_file["method"]

    binnedData = utils.bin_data(data_mat, binSize_mat, method_mat)
    binnedData_mat = mat_file["binnedData"]

    assert np.allclose(binnedData_mat, binnedData)

def test_get_rate_indices(get_rate_indices_data):
    """Test rate index quantization"""
    mat_file = get_rate_indices_data
    lambda_in_mat = mat_file["lambda_in"]
    lambdaRange_mat = mat_file["lambdaRange"]
    nRates_mat = mat_file["nRates"]

    v = utils.get_rate_indices(lambda_in_mat, lambdaRange_mat, nRates_mat)
    v_mat = mat_file["v"]

    assert np.allclose(v_mat, v)


def test_smooth_average(smooth_average_data):
    """Test smooth averaging with PCA"""
    mat_file = smooth_average_data
    hyper_params_mat = mat_file["HyperParams"][()]
    X_dict = {ix + 1: np.stack(v) if v.dtype == "O" else v[None, ...] 
              for ix, v in enumerate(mat_file["X_in"])}
    Ts_mat = mat_file["Ts"]
    X_bar_mat = np.stack(mat_file["X_bar"])
    
    # Extract parameters from hyperparameters
    soft_norm = hyper_params_mat["softNorm"]
    bin_size = hyper_params_mat["Delta"]  # Assuming bin_size = 1 for the test
    trial_dims = hyper_params_mat["nTrialDims"]
    neural_dims = hyper_params_mat["nNeuralDims"]
    condition_dims = hyper_params_mat["nCondDims"]

    X_bar = utils.smooth_average(
        X_dict, soft_norm, bin_size, Ts_mat, 
        trial_dims, neural_dims, condition_dims
    )

    assert np.allclose(X_bar_mat, np.stack([v for v in X_bar.values()]))

def test_fit_poisson_interp(fit_poisson_interp_data):
    """Test Poisson interpolation fitting"""
    mat_file = fit_poisson_interp_data
    S_mat = mat_file["S"]
    X1_mat = mat_file["X1"]
    X2_mat = mat_file["X2"]
    interp_options_mat = mat_file["InterpOptions"][()]
    default_alpha_mat = mat_file["defaultAlpha"]

    alpha_mat = mat_file["alpha"]
    
    # Extract max_iters and tolerance from InterpOptions
    max_iters = int(interp_options_mat["maxIters"])
    tolerance = float(interp_options_mat["stepTol"])

    alpha_py = utils.fit_poisson_interp(
        S_mat, X1_mat, X2_mat, max_iters, tolerance, default_alpha_mat
    )

    assert np.allclose(alpha_py, alpha_mat)

def test_recursion_less_t_prime(recursion_less_data):
    """Test recursion update (less t_prime case)"""
    mat_file = recursion_less_data

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

    Q_py = utils.recursion(
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

    assert np.allclose(Q_py, Q_out_mat)

def test_recursion_more_t_prime(recursion_more_data):
    """Test recursion update (more t_prime case)"""
    mat_file = recursion_more_data

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

    Q_py = utils.recursion(
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

    assert np.allclose(Q_py, Q_out_mat)

def test_get_time_indices(get_time_indices_data):
    """Test time index mapping"""
    mat_file = get_time_indices_data

    t_prime_mat = mat_file["t_prime"] - 1  # -1 for Python indexing
    T_prime_mat = mat_file["T_prime"]
    T_mat = mat_file["T"]
    Delta_mat = mat_file["Delta"]
    tau_prime_mat = mat_file["tau_prime"] - 1  # -1 for Python indexing
    causal_mat = mat_file["causal"]

    t_idx_mat = mat_file["tIdx"]

    t_idx_py, f_py = utils.get_time_indices(
        t_prime_mat,
        T_prime_mat,
        T_mat,
        Delta_mat,
        tau_prime_mat,
        causal_mat,
    )

    assert np.array_equal(t_idx_py, t_idx_mat - 1)  # -1 for Python indexing

def test_get_state_indices(get_time_indices_data, get_state_indices_data):
    """Test state index mapping"""
    # First get the time mapping function from get_time_indices
    mat_file_time = get_time_indices_data
    t_prime_mat = mat_file_time["t_prime"] - 1
    T_prime_mat = mat_file_time["T_prime"]
    T_mat = mat_file_time["T"]
    Delta_mat = mat_file_time["Delta"]
    tau_prime_mat = mat_file_time["tau_prime"] - 1
    causal_mat = mat_file_time["causal"]

    _, f_py = utils.get_time_indices(
        t_prime_mat, T_prime_mat, T_mat, Delta_mat, tau_prime_mat, causal_mat
    )

    # Now test get_state_indices
    mat_file_state = get_state_indices_data
    k_prime_hats_mat = mat_file_state["k_prime_hats"] - 1  # -1 for Python indexing
    K_mat = mat_file_state["K"] - 1  # -1 for Python indexing

    kIdx_mat = mat_file_state["kIdx"]
    kIdx_py = utils.get_state_indices(k_prime_hats_mat, f_py, K_mat)

    assert np.array_equal(kIdx_py, kIdx_mat - 1)  # -1 for Python indexing

def test_maximum_likelihood_no_restricted(maximum_likelihood_no_restricted_data):
    """Test maximum likelihood (no restricted conditions)"""
    mat_file = maximum_likelihood_no_restricted_data

    Q_mat = mat_file["Q_in"]
    tau_prime_mat = mat_file["tau_prime"] - 1
    first_idx_mat = mat_file["firstIdx"] - 1
    first_tau_prime_idx_mat = mat_file["firstTauPrimeIdx"] - 1
    restricted_conds_mat = mat_file["restrictedConds"]  # Empty

    c_hat_mat = mat_file["c_hat"]
    k_prime_hats_mat = mat_file["k_prime_hats"]

    c_hat_py, k_prime_hats_py = utils.maximum_likelihood(
        Q_mat,
        tau_prime_mat,
        first_idx_mat,
        first_tau_prime_idx_mat,
        restricted_conds_mat,
    )

    assert c_hat_py == c_hat_mat - 1
    assert np.array_equal(k_prime_hats_py, k_prime_hats_mat - 1)

def test_maximum_likelihood_with_restricted(maximum_likelihood_yes_restricted_data):
    """Test maximum likelihood (with restricted conditions)"""
    mat_file = maximum_likelihood_yes_restricted_data

    Q_mat = mat_file["Q_in"]
    tau_prime_mat = mat_file["tau_prime"] - 1
    first_idx_mat = mat_file["firstIdx"] - 1
    first_tau_prime_idx_mat = mat_file["firstTauPrimeIdx"] - 1
    restricted_conds_mat = [mat_file["restrictedConds"] - 1]  # With restriction

    c_hat_mat = mat_file["c_hat"]
    k_prime_hats_mat = mat_file["k_prime_hats"]

    c_hat_py, k_prime_hats_py = utils.maximum_likelihood(
        Q_mat,
        tau_prime_mat,
        first_idx_mat,
        first_tau_prime_idx_mat,
        restricted_conds_mat,
    )

    assert c_hat_py == c_hat_mat - 1
    assert np.array_equal(k_prime_hats_py, k_prime_hats_mat - 1)

def test_build_poisson():
    """Test Poisson lookup table construction"""
    # Test with typical parameters
    rate_range = (0.0, 2.0)  # spikes/sec
    num_rate_bins = 100
    bin_duration = 0.02  # 20ms bins
    min_prob = 1e-6
    
    log_likelihood, rate_centers = utils.build_poisson(
        rate_range, num_rate_bins, bin_duration, min_prob
    )
    
    # Check dimensions
    assert log_likelihood.shape[0] == num_rate_bins
    assert rate_centers.shape == (num_rate_bins, 1)
    
    # Check rate centers are correctly spaced (in spikes/sec, not spikes/bin)
    expected_rates = np.linspace(
        rate_range[0],
        rate_range[1], 
        num_rate_bins
    )[:, None]
    assert np.allclose(rate_centers, expected_rates)
    
    # Check log-likelihood properties
    assert np.all(np.isfinite(log_likelihood))
    assert log_likelihood.shape[1] == int(bin_duration * 1000) + 1

def test_flat_to_cond_state():
    """Test flat_to_cond_state conversion"""
    start_indices = np.array([0, 10, 25, 40])
    
    test_cases = [
        (5, (0, 5)),    # State 5 -> condition 0, local state 5
        (15, (1, 5)),   # State 15 -> condition 1, local state 5
        (30, (2, 5)),   # State 30 -> condition 2, local state 5
        (45, (3, 5)),   # State 45 -> condition 3, local state 5
    ]
    
    for flat_idx, (expected_cond, expected_local) in test_cases:
        cond, local = utils.flat_to_cond_state(flat_idx, start_indices)
        assert cond == expected_cond
        assert local == expected_local

def test_cond_state_to_flat():
    """Test cond_state_to_flat conversion"""
    start_indices = np.array([0, 10, 25, 40])
    
    test_cases = [
        ((0, 5), 5),    # Condition 0, local 5 -> flat 5
        ((1, 5), 15),   # Condition 1, local 5 -> flat 15
        ((2, 5), 30),   # Condition 2, local 5 -> flat 30
        ((3, 5), 45),   # Condition 3, local 5 -> flat 45
    ]
    
    for (cond, local), expected_flat in test_cases:
        flat_idx = utils.cond_state_to_flat(cond, local, start_indices)
        assert flat_idx == expected_flat

def test_state_index_conversions_roundtrip():
    """Test roundtrip conversion between flat and (condition, local) indices"""
    start_indices = np.array([0, 10, 25, 40])
    
    # Test round-trip conversions
    for cond in range(len(start_indices)):
        for local in range(5):
            flat_idx = utils.cond_state_to_flat(cond, local, start_indices)
            cond_back, local_back = utils.flat_to_cond_state(flat_idx, start_indices)
            assert cond_back == cond
            assert local_back == local