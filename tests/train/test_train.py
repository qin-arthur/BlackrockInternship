import os

import numpy as np
# import pandas as pd
import pytest
import scipy.io
# import h5py

from ..definitions import DIR
from scripts.mc_maze_example.mc_maze_config import mc_maze_config
from brn.mint.gauss_filt import gauss_filt
from brn.mint.process_kinematics import process_kinematics
from brn.mint.fit_trajectories import fit_trajectories
from brn.mint.smooth_average import smooth_average
from brn.mint.get_rate_indices import get_rate_indices
from brn.mint.bin_data import bin_data


# --------------------------------------------------------


@pytest.fixture
def config_tuple():
    [settings, hyper_params] = mc_maze_config(True)
    return settings, hyper_params


@pytest.fixture()
def load_mat_Post_Training_50_tuple():
    fpath = os.path.join(DIR, "Post_training_50.mat")
    # f = h5py.File(fpath, 'r')
    mat_file = scipy.io.loadmat(fpath, squeeze_me=True)
    return mat_file


@pytest.fixture()
def load_mat_50_post_fit_trajectories_tuple():
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "fit_trajectories_variables.mat"), squeeze_me=True
    )
    return mat_file


@pytest.fixture()
def load_mat_50_post_gauss_filt_tuple():
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "gauss_filt_variables.mat"), squeeze_me=True
    )
    return mat_file


@pytest.fixture()
def load_mat_process_kinematics_tuple():
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "process_kinematics_variables.mat"), squeeze_me=True
    )
    return mat_file


@pytest.fixture()
def load_mat_50_smooth_average_tuple():
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "smooth_average_variables.mat"), squeeze_me=True
    )
    return mat_file


@pytest.fixture()
def load_mat_50_get_rate_indices_tuple():
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "get_rate_indices_variables.mat"), squeeze_me=True
    )
    return mat_file


@pytest.fixture()
def load_mat_50_bin_data_tuple():
    mat_file = scipy.io.loadmat(
        os.path.join(DIR, "binned_data_variables.mat"), squeeze_me=True
    )
    return mat_file


# -------------------------------------------------------


def test_mc_maze_config(config_tuple, load_mat_Post_Training_50_tuple):
    settings, hyper_params = config_tuple
    mat_file = load_mat_Post_Training_50_tuple

    settings_mat = np.array(mat_file["Settings"])
    hyper_params_mat = np.array(mat_file["HyperParams"])

    assert settings["task"] == settings_mat["task"]
    assert settings["Ts"] == settings_mat["Ts"]
    assert np.allclose(settings["trialAlignment"], settings_mat["trialAlignment"][()])
    assert np.allclose(settings["testAlignment"], settings_mat["testAlignment"][()])

    assert np.allclose(
        hyper_params["trajectoriesAlignment"],
        hyper_params_mat["trajectoriesAlignment"][()],
    )
    assert np.isnan(hyper_params_mat["nNeuralDims"][()]) & np.isnan(
        hyper_params["nNeuralDims"]
    )

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
        assert hyper_params[hyper] == hyper_params_mat[hyper]


def test_fit_trajectories(load_mat_50_post_fit_trajectories_tuple):
    mat_file = load_mat_50_post_fit_trajectories_tuple
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

    assert np.allclose(Omega_plus_mat, np.stack([_ for _ in Omega_plus.values()]))
    assert np.allclose(Phi_plus_mat, np.stack([_ for _ in Phi_plus.values()]))
    assert MiscParams_mat.tolist() == MiscParams["kin_labels"]


def test_gauss_filt(load_mat_50_post_gauss_filt_tuple):
    mat_file = load_mat_50_post_gauss_filt_tuple
    binSize_mat = mat_file["binSize"]
    sigma_mat = mat_file["sigma"]
    spikes_mat = mat_file["spikes"]
    filtSpikes_mat = mat_file["filtSpikes"]

    filtSpikes_py = gauss_filt(spikes_mat, sigma_mat, binSize_mat)

    assert np.allclose(filtSpikes_py, filtSpikes_mat)


def test_process_kinematics(load_mat_process_kinematics_tuple):
    Z_mat = np.stack(load_mat_process_kinematics_tuple["Z"])
    settings_mat = load_mat_process_kinematics_tuple["Settings"]
    Z_out_mat = np.stack(load_mat_process_kinematics_tuple["Z_out"])
    labels_mat = load_mat_process_kinematics_tuple["labels"]
    Z_py, labels_py = process_kinematics(Z_mat, settings_mat[()])

    assert labels_py == labels_mat.tolist()
    assert np.allclose(Z_py, Z_out_mat)


def test_smooth_average(load_mat_50_smooth_average_tuple):
    mat_file = load_mat_50_smooth_average_tuple
    hyper_params_mat = mat_file["HyperParams"][()]
    X_dict = {ix + 1: np.stack(v) if v.dtype == "O" else v[None, ...] for ix, v in enumerate(mat_file["X_in"])}
    Ts_mat = mat_file["Ts"]
    X_bar_mat = np.stack(mat_file["X_bar"])

    X_bar = smooth_average(X_dict, hyper_params_mat, Ts_mat)

    assert np.allclose(X_bar_mat, np.stack([v for v in X_bar.values()]))


def test_bin_data(load_mat_50_bin_data_tuple):
    mat_file = load_mat_50_bin_data_tuple
    data_mat = mat_file["data"]
    binSize_mat = mat_file["binSize"]
    method_mat = mat_file["method"]

    binnedData = bin_data(data_mat, binSize_mat, method_mat)
    binnedData_mat = mat_file["binnedData"]

    assert np.allclose(binnedData_mat, binnedData)


def test_get_rate_indices(load_mat_50_get_rate_indices_tuple):
    mat_file = load_mat_50_get_rate_indices_tuple
    lambda_in_mat = mat_file["lambda_in"]
    lambdaRange_mat = mat_file["lambdaRange"]
    nRates_mat = mat_file["nRates"]

    v = get_rate_indices(lambda_in_mat, lambdaRange_mat, nRates_mat)
    v_mat = mat_file["v"]

    assert np.allclose(v_mat, v)


# -------------------************---------------------
