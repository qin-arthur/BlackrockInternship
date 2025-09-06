import os
import time
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

from brn.mint.decoder import MINTDecoder, MINTSettings
from brn.mint.utils import process_kinematics, bin_data
from brn.mint.generic_config import generic_config





#Requires DANDI Dataset folder "000128" to be in Downloads directory
def get_trial_data(settings: dict, split: str = "train", n_trials: int | None = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Helper function to get trial data from the MC Maze dataset.

    :param settings: dict with key "trialAlignment" which is a list of integers representing the indices
      for each trial relative to its move_onset_time.
    :param split: "train" or "val"
    :param n_trials: Number of trials to return, if None, returns all trials
    :return: S, Z, condition
        S: ndarray of shape (n_trials, n_neurons, n_time_steps), dtype bool with 1 at spike time, 0 otherwise . #np.uint64(50321)
        Z: ndarray of shape (n_trials, 4 (pos_x, pos_y, vel_x, vel_y), n_time_steps), dtype float
        condition: ndarray of shape (n_trials,), dtype int with condition indices
    """
    datadir = os.getenv("MC_MAZE_DATASET_PATH", str(Path.home() / "Downloads" / "000128"))
    train_path = Path(datadir) / "sub-Jenkins" / "sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb"
    test_path = Path(datadir) / "sub-Jenkins" / "sub-Jenkins_ses-full_desc-test_ecephys.nwb"

    # Load dataset
    import pynwb
    with pynwb.NWBHDF5IO(test_path if split == "test" else train_path, "r") as io:
        read_nwbfile = io.read()
        # metadata fields: devices, electrode_groups, electrodes, session_description, subject, etc.

        # Find the trial info and keep only the subset we want
        trial_info = read_nwbfile.intervals["trials"].to_dataframe()
        trial_info = trial_info[trial_info["split"] == split]

        n_trials = n_trials or len(trial_info)

        # Find unique conditions
        if split == "test":
            condition = -1 * np.ones(n_trials, dtype=int)
        else:
            conds = trial_info.set_index(["trial_type", "trial_version"]).index.tolist()
            cond_list = trial_info.set_index(["trial_type", "trial_version"]).index.unique().tolist()
            cond_list.sort()
            condition = np.array([cond_list.index(_) + 1 for _ in conds[:n_trials]])

        # Get a timestamp vector for all behavior data
        behav_dict = read_nwbfile.processing["behavior"].data_interfaces
        assert np.array_equal(behav_dict["hand_pos"].timestamps[:], behav_dict["hand_vel"].timestamps[:])
        all_ts = behav_dict["hand_pos"].timestamps[:]

        # Find the index where each trial's move period starts
        trial_onsets_beh = np.searchsorted(all_ts, trial_info[:n_trials]["move_onset_time"])

        # Take per-trial data from position and velocity arrays.
        trial_inds = settings["trialAlignment"]
        all_trial_inds = trial_onsets_beh[:, None] + trial_inds[None, :]
        all_trial_ts = np.take(all_ts, all_trial_inds, axis=0)

        all_pos = np.take(behav_dict["hand_pos"].data, all_trial_inds, axis=0)
        all_vel = np.take(behav_dict["hand_vel"].data, all_trial_inds, axis=0)
        Z = np.concatenate((all_pos, all_vel), axis=-1)
        Z = np.transpose(Z, (0, 2, 1))  # Shape -> (n_trials, 4, n_steps)

        # Create a boolean array for spikes with the same shape for trials and times as the behaviour data.
        n_neurons = read_nwbfile.units["spike_times"].data.shape[0]
        n_steps = len(trial_inds)
        S = np.zeros((n_trials, n_neurons, n_steps), dtype=bool)
        # For each neuron, for each trial, find the spike times and set the corresponding indices to True
        for unit_ix in range(n_neurons):
            spike_times = read_nwbfile.units["spike_times"][unit_ix]
            for trial_ix, trial_tvec in enumerate(all_trial_ts):
                # Find the spike times that fall within the trial time vector
                trial_spike_times = spike_times[np.logical_and(spike_times >= trial_tvec[0], spike_times <= trial_tvec[-1])]
                spike_inds = np.searchsorted(trial_tvec, trial_spike_times)
                # Set the corresponding indices in S to True
                S[trial_ix, unit_ix, spike_inds] = True

    return S, Z, condition



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



[settings, hyper_params] = mc_maze_config(True)
S_all, Z_all, condition_all = get_trial_data(settings, split="train")



settings = MINTSettings(
    task='mc_maze', 
    data_path="C:/Users/aqin/Downloads/000128/sub-Jenkins/",
    results_path="C:/Users/aqin/Downloads/results",
    bin_size=20,
    observation_window=300,
    causal=True,
    trial_alignment=np.arange(-800, 901, 1),
    test_alignment=np.arange(-250, 451, 1),
    trajectories_alignment=np.arange(-500, 701, 1),
    gaussian_sigma=30,
    neural_dims=np.nan,
    condition_dims=21,
    trial_dims=1,
    min_lambda=1.0,
    sampling_period=0.001,
    soft_norm=5.0,
    min_prob=1e-6,
    min_rate=0.0,
    interp_mode=2,
    interp_max_iters=10,
    interp_tolerance=0.01,
    num_rate_bins=2000
)




def calculate_r2(y_true, y_pred):
    """Compute R² score between true and predicted arrays."""
    
    
    # Remove any time bins where decode is NaN
    nan_mask = np.any(np.isnan(y_pred), axis=0)
    y_true = y_true[:, ~nan_mask]
    y_pred = y_pred[:, ~nan_mask]

    # Residual sum of squares
    SS_res = np.sum((y_true - y_pred) ** 2, axis=1)

    # Total sum of squares
    mean_true = np.mean(y_true, axis=1, keepdims=True)
    SS_tot = np.sum((y_true - mean_true) ** 2, axis=1)

    # Coefficient of determination
    R2 = 1 - SS_res / SS_tot

    return R2

decoder = MINTDecoder(settings)
t0 = time.perf_counter()
decoder.fit(spikes=S_all, behavior=Z_all, cond_ids=condition_all)
fit_secs = time.perf_counter() - t0
print(f"Fitting took {fit_secs:.3f} s")

#Post-predict function prediction stuff starts here
Z_val = process_kinematics(Z_val, decoder.settings.trial_alignment)
test_buffer = np.array((-decoder.settings.observation_window + 1, 0))
if not decoder.settings.causal:
    test_buffer = test_buffer + np.round(
        (decoder.settings.observation_window + decoder.settings.bin_size) / 2
    )

buff_align_start = decoder.settings.test_alignment[0] + test_buffer[0]
buff_align_end = decoder.settings.test_alignment[-1] + test_buffer[1]
buffered_alignment = np.arange(buff_align_start, buff_align_end + 1)
t_mask = np.isin(decoder.settings.trial_alignment, buffered_alignment)
S_val = S_val[:, :, t_mask]
Z_val = Z_val[:, :, t_mask]

t1 = time.perf_counter()
X_hat, Z_hat = decoder.predict(S_val)
pred_secs = time.perf_counter() - t1
print(f"Predict function took {pred_secs:.3f} s")

not_buff_mask = np.isin(buffered_alignment, decoder.settings.test_alignment)
Z_val = Z_val[..., not_buff_mask]
Z_hat = Z_hat[..., not_buff_mask]
X_hat = X_hat[..., not_buff_mask]

# Convert neural state estimate to spikes/second
X_hat /= (decoder.settings.bin_size * decoder.settings.sampling_period)

# Store estimates
# Note: previously each value was a list of arrays, now just an array.
estimates = {
    "Z": Z_val,
    "Z_hat": Z_hat,
    "X_hat": X_hat,
}







eval_bin_size = 5

# unpack
behavior               = estimates['Z']
behavior_estimate      = estimates['Z_hat']
neural_state_estimate  = estimates['X_hat']

# bin each trial with a moving‐window mean
Z_binned     = [bin_data(z, eval_bin_size, method='mean') for z in behavior]
Z_hat_binned = [bin_data(zhat, eval_bin_size, method='mean') for zhat in behavior_estimate]

# stitch trials together along the time axis
Z_concat     = np.hstack(Z_binned)
Z_hat_concat = np.hstack(Z_hat_binned)

# compute R² per feature
R2 = calculate_r2(Z_concat, Z_hat_concat)
avg_R2 = np.mean(R2)

print("R2 per feature: " + str(R2))
print("Final R2: " + str(avg_R2))

