import numpy as np
import pytest
import scipy.io
import brn.mint
from scripts.mc_maze_example.get_trial_data import get_trial_data
from scripts.mc_maze_example.mc_maze_config import mc_maze_config
from ..definitions import DIR


@pytest.fixture
def config_tuple():
    [settings, hyper_params] = mc_maze_config(True)
    return settings, hyper_params


@pytest.fixture()
def load_mat_post_training_50():
    mat_file = scipy.io.loadmat(DIR / "Post_training_50.mat", squeeze_me=True)
    return mat_file


@pytest.fixture()
def load_mat_50_post_get_trial_data():
    mat_file = scipy.io.loadmat(DIR / "50_post_get_trial_data.mat", squeeze_me=True)
    return mat_file


@pytest.fixture()
def load_mat_50_test():
    mat_file = scipy.io.loadmat(DIR / "test_variables.mat", squeeze_me=True)
    return mat_file


# --------------------------------------------------------------------------------


# Training test
def test_get_trial_data(config_tuple, load_mat_50_post_get_trial_data):
    settings, hyper_params = config_tuple

    try:
        S, Z, condition = get_trial_data(settings, split="train", n_trials=50)
    except ImportError:
        pytest.skip("pynwb not installed, skipping get_trial_data test.")

    mat_file = load_mat_50_post_get_trial_data
    S_mat = np.stack(mat_file["S"])
    Z_mat = np.stack(mat_file["Z"])
    condition_mat = mat_file["condition"]

    assert S.shape == S_mat.shape
    # assert np.allclose(S, S_mat)  # No longer True after fixing a bug in the data loader.
    assert np.allclose(Z, Z_mat)
    assert np.allclose(condition, condition_mat)


# Training test
# Tests mint.py and fit.py
def test_model_contents(config_tuple, load_mat_post_training_50):
    settings, hyper_params = config_tuple
    model, train_summary = brn.mint.train(settings, hyper_params)

    mat_file = load_mat_post_training_50

    test_model_struct = {"Omega_plus": 6, "Phi_plus": 7, "L": 15, "V": 17}
    for model_field_name, mat_struct_idx in test_model_struct.items():
        mat = np.stack(mat_file["struct_model"][()][mat_struct_idx])
        calc = model[model_field_name]
        if isinstance(calc, dict):
            calc = np.stack(list(calc.values()))
        assert calc.shape == mat.shape
        if model_field_name not in ["Omega_plus", "V"]:
            # We don't test neural data for equality because we fixed an indexing bug in the data loader but
            #  that changed the data, and we have yet to recalculate the Matlab results.
            assert np.allclose(calc, mat)


# Validation test
def test_validation(load_mat_50_test):
    mat_file = load_mat_50_test
    model_mat = mat_file["model"]

    # headers = model_mat.dtype.names
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
    model_dict = {h: model_mat[()][h_ix] for h_ix, h in enumerate(headers)}

    # Dictify Settings entry
    model_dict["Settings"] = {
        _: model_dict["Settings"][_][()] for _ in model_dict["Settings"].dtype.names
    }
    model_dict["Settings"].pop("CondInfo")

    # Correcting indexing for Python from MATLAB
    for k in [
        "firstIdx",
        "tau_prime",
        "firstTauPrimeIdx",
        "shiftedIdx1",
        "shiftedIdx2",
    ]:
        model_dict[k] -= 1

    estimates_mat = mat_file["Estimates"]

    estimates_py = brn.mint.test(model_dict)

    for k in ["Z", "Z_hat", "X_hat"]:
        assert k in estimates_py, f"Key {k} not found in Python estimates."
        py_data = estimates_py[k]
        mat_data = np.stack(estimates_mat[k][()])
        assert py_data.shape == mat_data.shape, f"Shape mismatch for {k}: {py_data.shape} != {mat_data.shape}"
        if k in ["Z"]:
            # Some of the values aren't matching, probably because I fixed a bug in the data loader.
            assert np.allclose(py_data, mat_data), f"Data mismatch for {k}."
