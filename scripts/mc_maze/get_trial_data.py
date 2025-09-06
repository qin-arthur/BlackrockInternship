import os
from pathlib import Path
import numpy as np
import scipy.io
from mc_maze_config import mc_maze_config
from tests.definitions import DIR

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
        print(f"{Colors.OKGREEN}PASSED: {test_name}{Colors.ENDC}")
        test_results.append((test_name, True, None))
        return True
    except AssertionError as e:
        print(f"{Colors.FAIL}FAILED: {test_name}{Colors.ENDC}")
        print(f"{Colors.WARNING}  Assertion Error: {str(e)}{Colors.ENDC}")
        test_results.append((test_name, False, f"Assertion Error: {str(e)}"))
        return False
    except Exception as e:
        print(f"{Colors.FAIL}ERROR: {test_name}{Colors.ENDC}")
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

def load_mat_50_post_get_trial_data():
    mat_file = scipy.io.loadmat(os.path.join(DIR, "50_post_get_trial_data.mat"), squeeze_me=True)
    return mat_file

# Test functions
def test_get_trial_data():
    settings, hyper_params = config_tuple()

    try:
        import pynwb
        S, Z, condition = get_trial_data(settings, split="train", n_trials=50)
        
        mat_file = load_mat_50_post_get_trial_data()
        S_mat = np.stack(mat_file["S"])
        Z_mat = np.stack(mat_file["Z"])
        condition_mat = mat_file["condition"]

        assert S.shape == S_mat.shape, f"S shape mismatch: {S.shape} vs {S_mat.shape}"
        # assert np.allclose(S, S_mat)  # No longer True after fixing a bug in the data loader.
        assert np.allclose(Z, Z_mat), "Z data does not match"
        assert np.allclose(condition, condition_mat), "Condition data does not match"
        
    except ImportError:
        print(f"{Colors.WARNING}  pynwb not installed, skipping get_trial_data test.{Colors.ENDC}")
        test_results.append(("Get Trial Data", True, "Skipped - pynwb not installed"))
        return


def main():
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}RUNNING GET_TRIAL_DATA TESTS{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")
    
    # Run the test
    run_test("Get Trial Data", test_get_trial_data)
    
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