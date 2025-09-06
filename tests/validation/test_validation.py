import os

import pytest
import numpy as np
import scipy.io

from ..definitions import DIR
from brn.mint.ind2ck import ind2ck
from brn.mint.ck2ind import ck2ind
from brn.mint.maximum_likelihood import maximum_likelihood
from brn.mint.fit_poisson_interp import fit_poisson_interp
from brn.mint.recursion import recursion
from brn.mint.estimate_states import estimate_states
from brn.mint.get_state_indices import get_state_indices
from brn.mint.get_time_indices import get_time_indices
from brn.mint.predict import predict


@pytest.fixture()
def load_mat_ind2ck_tuple():
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "ind2ck_variables.mat"), squeeze_me=True
    )
    return mat_file


@pytest.fixture()
def load_mat_ck2ind_tuple():
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "ck2ind_variables.mat"), squeeze_me=True
    )
    return mat_file


@pytest.fixture()
def load_mat_maximum_likelihood_no_restrictedCond_tuple():
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "maximum_likelihood_variables_no_restrictedCond.mat"),
        squeeze_me=True,
    )
    return mat_file


@pytest.fixture()
def load_mat_maximum_likelihood_yes_restrictedCond_tuple():
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "maximum_likelihood_variables_yes_restrictedCond.mat"),
        squeeze_me=True,
    )
    return mat_file


@pytest.fixture()
def load_mat_fit_poisson_interp_tuple():
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "fit_poisson_interp_variables.mat"), squeeze_me=True
    )
    return mat_file


@pytest.fixture()
def load_mat_recursion_less_tuple():
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "recursion_variables_t_prime_less.mat"), squeeze_me=True
    )
    return mat_file


@pytest.fixture()
def load_mat_recursion_more_tuple():
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "recursion_variables_t_prime_more.mat"), squeeze_me=True
    )
    return mat_file


@pytest.fixture()
def load_mat_estimate_states_tuple():
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "estimate_states_variables.mat"), squeeze_me=True
    )
    return mat_file


@pytest.fixture()
def load_mat_get_state_indices_tuple():
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "get_state_indices_variables.mat"), squeeze_me=True
    )
    return mat_file


@pytest.fixture()
def load_mat_get_time_indices_tuple():
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "get_time_indices_variables.mat"), squeeze_me=True
    )
    return mat_file


@pytest.fixture()
def load_mat_predict_tuple():
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "predict_variables.mat"), squeeze_me=True
    )
    return mat_file


# ---------------------------------------------------------------------------------------


def test_ind2ck(load_mat_ind2ck_tuple):
    mat_file = load_mat_ind2ck_tuple

    i_mat = mat_file["i"]
    first_idx_mat = mat_file["firstIdx"]
    c_mat = mat_file["c"]
    k_mat = mat_file["k"]

    c_py, k_py = ind2ck(i_mat, first_idx_mat)

    assert c_py == c_mat - 1 and k_py == k_mat - 1


def test_ck2ind(load_mat_ck2ind_tuple):
    mat_file = load_mat_ck2ind_tuple

    c_mat = mat_file["c"] - 1
    k_mat = mat_file["k"] - 1
    first_idx_mat = mat_file["firstIdx"] - 1
    i_mat = mat_file["i"]

    i_py = ck2ind(c_mat, k_mat, first_idx_mat)

    assert np.any(i_py == i_mat - 1)


def test_maximum_likelihood_no_restrictedConds(
    load_mat_maximum_likelihood_no_restrictedCond_tuple,
):
    mat_file = load_mat_maximum_likelihood_no_restrictedCond_tuple

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
    )


def test_maximum_likelihood_yes_restrictedConds(
    load_mat_maximum_likelihood_yes_restrictedCond_tuple,
):
    mat_file = load_mat_maximum_likelihood_yes_restrictedCond_tuple

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

    assert (c_hat_py == c_hat_mat - 1) & np.any(k_prime_hats_py == k_prime_hats_mat - 1)


def test_fit_poisson_interp(load_mat_fit_poisson_interp_tuple):
    mat_file = load_mat_fit_poisson_interp_tuple
    S_mat = mat_file["S"]
    X1_mat = mat_file["X1"]
    X2_mat = mat_file["X2"]
    interp_options_mat = mat_file["InterpOptions"]
    default_alpha_mat = mat_file["defaultAlpha"]

    alpha_mat = mat_file["alpha"]

    alpha_py = fit_poisson_interp(
        S_mat, X1_mat, X2_mat, interp_options_mat, default_alpha_mat
    )

    assert np.allclose(alpha_py, alpha_mat)


def test_recursion_less_t_prime(load_mat_recursion_less_tuple):
    mat_file = load_mat_recursion_less_tuple

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

    assert np.allclose(Q_py, Q_out_mat)


def test_recursion_more_t_prime(load_mat_recursion_more_tuple):
    mat_file = load_mat_recursion_more_tuple

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

    assert np.allclose(Q_py, Q_out_mat)


def test_estimate_states(
    load_mat_estimate_states_tuple, load_mat_get_time_indices_tuple
):
    mat_file = load_mat_estimate_states_tuple

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
    f_py = get_f(load_mat_get_time_indices_tuple)

    X_hat_py, Z_hat_py, C_hat_py, K_hat_py, Alpha_hat_py = estimate_states(
        Q_in_mat, S_curr_mat, f_py, model_dict
    )
    X_hat_py = X_hat_py.T
    Z_hat_py = Z_hat_py.T
    C_hat_py = C_hat_py.T
    K_hat_py = K_hat_py.T
    Alpha_hat_py = Alpha_hat_py.T

    assert np.allclose(X_hat_py, X_hat_mat) and np.allclose(Z_hat_py, Z_hat_mat)


def test_get_time_and_state_indices(
    load_mat_get_time_indices_tuple, load_mat_get_state_indices_tuple
):
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

    # ---------get_state_indices
    mat_file_state = load_mat_get_state_indices_tuple

    k_prime_hats_mat_state = (
        mat_file_state["k_prime_hats"] - 1
    )  # -1 for Python indexing
    f_mat_state = mat_file_state["f"]
    K_mat_state = mat_file_state["K"] - 1  # -1 for Python indexing

    kIdx_mat_state = mat_file_state["kIdx"]
    kIdx_py_state = get_state_indices(k_prime_hats_mat_state, f_py_time, K_mat_state)

    assert np.any(t_idx_py_time == t_idx_mat_time_mat - 1) & np.any(
        kIdx_py_state == kIdx_mat_state - 1
    )  # -1 for Python Indexing


def test_predict(load_mat_predict_tuple):
    mat_file = load_mat_predict_tuple

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
    )


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
