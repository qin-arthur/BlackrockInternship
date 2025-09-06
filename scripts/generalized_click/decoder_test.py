import numpy as np
from pathlib import Path
from typing import Union
from brn.mint.model import MINT, MINTSettings

settings = MINTSettings(
task="mc_maze",
fs=50.0,
obs_window=500,
min_lambda=1.0,
n_rates=2000,
min_rate=0.0,
max_rate=20,
min_prob=1e-6,
interp_mode=3,
interp_max_iters=10,
interp_tolerance=0.01
)

def make_synthetic_mint_unit_test(
    n_train_trials: int = 6,
    n_test_trials: int = 4,
    n_time_bins: int = 120,
    n_neurons: int = 192,
    n_conditions: int = 8,
    bin_ms: int = 20,
    seed: int = 0,
):
    """
    Returns:
      train_features: (n_train_trials, n_time_bins, n_neurons)  int spike counts
      train_behavior: (n_train_trials, n_time_bins, 4)          [x, y, vx, vy]
      train_conditions: (n_train_trials,)                       ints in [0, n_conditions)
      test_features:  (n_test_trials, n_time_bins, n_neurons)
      test_behavior:  (n_test_trials, n_time_bins, 4)
    """
    rng = np.random.default_rng(seed)
    dt = bin_ms / 1000.0

    # Neuron tuning: preferred directions, baselines, gains
    angles = rng.uniform(0, 2*np.pi, size=n_neurons)
    PD = np.stack([np.cos(angles), np.sin(angles)], axis=1)          # (n_neurons, 2)
    baseline_hz = rng.uniform(3.0, 12.0, size=n_neurons)             # Hz
    gain = rng.uniform(8.0, 25.0, size=n_neurons)                    # Hz per (unit vel dot PD)

    def make_block(n_trials, start_seed_offset=0):
        r = np.random.default_rng(seed + start_seed_offset)
        feats = np.zeros((n_trials, n_time_bins, n_neurons), dtype=np.int16)
        behav = np.zeros((n_trials, n_time_bins, 4), dtype=np.float32)
        conds = np.zeros((n_trials,), dtype=np.int16)

        # Target directions equally spaced on the unit circle
        target_dirs = np.stack([
            np.cos(np.linspace(0, 2*np.pi, n_conditions, endpoint=False)),
            np.sin(np.linspace(0, 2*np.pi, n_conditions, endpoint=False))
        ], axis=1)  # (n_conditions, 2)

        # Smooth progress profile 0→1 (minimum-jerk-ish)
        t = np.linspace(0, 1, n_time_bins, endpoint=True)
        s = 3*t**2 - 2*t**3  # smooth step
        ds_dt = np.gradient(s, dt)

        for i in range(n_trials):
            c = i % n_conditions
            conds[i] = c
            target = target_dirs[c]
            distance = r.uniform(8.0, 14.0)  # arbitrary distance units

            # Position over time (x,y) with small noise
            pos = (s[:, None] * distance * target[None, :])
            pos += r.normal(0, 0.05, size=pos.shape)

            # Velocity from finite differences (ẋ,ẏ)
            vel = np.gradient(pos, dt, axis=0)

            # Behavior: [x, y, vx, vy]
            bh = np.concatenate([pos, vel], axis=1)  # (T, 4)
            behav[i] = bh.astype(np.float32)

            # Neural rates: baseline + gain * (vel ⋅ PD) with ReLU & clipping
            # vel: (T,2), PD: (N,2) -> proj: (T,N)
            proj = vel @ PD.T
            rate_hz = baseline_hz[None, :] + gain[None, :] * proj
            rate_hz = np.maximum(rate_hz, 0.5)          # floor to keep Poisson well-defined
            rate_hz = np.clip(rate_hz, 0.5, 100.0)      # keep rates sane

            lam = rate_hz * dt                           # expected spikes per bin
            feats[i] = r.poisson(lam).astype(np.int16)

        return feats, behav, conds

    train_features, train_behavior, train_conditions = make_block(n_train_trials, start_seed_offset=0)
    test_features,  test_behavior,  _                = make_block(n_test_trials,  start_seed_offset=999)

    # Sanity checks on shapes
    assert train_features.shape == (n_train_trials, n_time_bins, n_neurons)
    assert train_behavior.shape == (n_train_trials, n_time_bins, 4)
    assert train_conditions.shape == (n_train_trials,)
    assert test_features.shape  == (n_test_trials,  n_time_bins, n_neurons)
    assert test_behavior.shape  == (n_test_trials,  n_time_bins, 4)

    return train_features, train_behavior, train_conditions, test_features, test_behavior


train_features, train_behav, train_conditions, test_features, test_behav = make_synthetic_mint_unit_test()

# Initialize decoder
decoder = MINT(settings)

decoder.fit(
    train_features,
    train_behav,
    train_conditions
)

import numpy as np
import time

# ---- helpers ----
CHANNELS = ["x_pos", "y_pos", "x_vel", "y_vel"]

def r2_per_channel(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute R² per channel for a single trial.
    y_true, y_pred: (T, K) with K behavioral variables.
    Returns: (K,) with NaN if SS_tot == 0 or not enough valid samples.
    """
    assert y_true.ndim == 2 and y_pred.ndim == 2
    T = min(len(y_true), len(y_pred))
    if T == 0:
        return np.full(y_true.shape[1], np.nan, dtype=float)

    yt = y_true[:T]
    yp = y_pred[:T]

    K = min(yt.shape[1], yp.shape[1])
    r2 = np.full(K, np.nan, dtype=float)

    # Channel-wise R² (ignore NaNs independently per channel)
    for k in range(K):
        valid = (~np.isnan(yt[:, k])) & (~np.isnan(yp[:, k]))
        if np.count_nonzero(valid) < 2:
            continue
        yk = yt[valid, k]
        pk = yp[valid, k]
        ss_res = np.sum((yk - pk) ** 2)
        ss_tot = np.sum((yk - np.mean(yk)) ** 2)
        if ss_tot > 0:
            r2[k] = 1.0 - ss_res / ss_tot
        # else: leave as NaN (constant ground-truth channel → undefined R²)
    return r2

# ---- evaluation loop ----
count = 0
time_sum = 0.0
trial_r2s = []  # list of (4,) arrays

print("test_features:", test_features.shape)
print("test_behavior:", test_behav.shape)

for idx, (feat, beh) in enumerate(zip(test_features, test_behav), start=1):
    # feat: (T, N), beh: (T, 4)
    t0 = time.perf_counter()
    X_hat, Z_hat, C_hat, K_hat, Alpha_hat = decoder.predict(feat)
    dt = time.perf_counter() - t0

    # Score this trial
    r2_vec = r2_per_channel(beh, Z_hat)   # (4,)
    trial_r2s.append(r2_vec)

    time_sum += dt
    count += 1
    # Print timing and R² for this trial (compact)
    r2_str = ", ".join(f"{name}: {val:.3f}" if np.isfinite(val) else f"{name}: nan"
                       for name, val in zip(CHANNELS, r2_vec))
    print(f"Trial {idx}: {dt:.3f}s | R² -> {r2_str}")

# ---- summary ----
trial_r2s = np.vstack(trial_r2s)  # (n_trials, 4)
per_feature_mean = np.nanmean(trial_r2s, axis=0)  # mean over trials, per channel
overall_mean = np.nanmean(trial_r2s)              # mean over all trials & channels
avg_time = time_sum / max(count, 1)

print("\n=== R² Summary (mean across trials) ===")
for name, val in zip(CHANNELS, per_feature_mean):
    print(f"{name:>6}: {val:.3f}")
print(f"Overall average R²: {overall_mean:.3f}")
print(f"Trial average time: {avg_time:.3f}s")
