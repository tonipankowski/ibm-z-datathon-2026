"""Train the AquaWatch anomaly detector on normal-operation sensor data.

Usage:
    python -m ai.train --data path/to/sensors.csv [--out ai/model.joblib] [--window 10]

The saved joblib bundle is loaded by ai/detect.py, which also imports
build_features() from here so training and live detection use identical features.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# --- Column names (edit here if Role 1's simulator uses different names) ---
TIMESTAMP_COL = "timestamp"
SCENARIO_COL = "scenario"
LEAK_PIPE_COL = "leak_pipe"  # ground truth, evaluation only, never a feature
NORMAL_SCENARIO = "normal"

# EPANET Net1 ids: P_<junction> pressures, F_<pipe> flows.
PRESSURE_COLS: list[str] = [f"P_{j}" for j in (10, 11, 12, 13, 21, 22, 23, 31, 32)]
FLOW_COLS: list[str] = [f"F_{p}" for p in (10, 11, 12, 21, 22, 31, 110, 111, 112, 113, 121, 122)]
OPTIONAL_SENSOR_COLS: list[str] = ["F_9"]  # pump flow, used only if the CSV has it
SENSOR_COLS: list[str] = PRESSURE_COLS + FLOW_COLS
REQUIRED_COLS: list[str] = [TIMESTAMP_COL, *SENSOR_COLS, SCENARIO_COL]

DEFAULT_WINDOW = 10
DEFAULT_MODEL_PATH = "ai/model.joblib"

# --- Time-of-day baseline ---
SECONDS_PER_DAY = 86400
SLOT_SECONDS = 300  # 5-minute slots, matching the simulator's sampling step
# Std floors (simulated data has no sensor noise, so raw stds can be ~0):
PRESSURE_STD_FLOOR = 0.1     # m of head, ~0.1% of a 100 m pressure transducer range
FLOW_STD_FLOOR_REL = 0.01    # 1% of the sensor's typical |flow| (installed meter accuracy)
FLOW_STD_FLOOR_ABS = 1e-5    # m3/s (0.01 L/s), for pipes with near-zero flow

# --- Model hyperparameters ---
N_ESTIMATORS = 200
CONTAMINATION = 0.01
RANDOM_STATE = 42


def load_data(csv_path: str | Path) -> pd.DataFrame:
    """Read the simulator CSV, validate its columns and sort it by time.

    Raises:
        ValueError: if any required column is missing.
    """
    df = pd.read_csv(csv_path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Numeric timestamps (e.g. seconds since start) are kept as-is; strings are parsed.
    if not pd.api.types.is_numeric_dtype(df[TIMESTAMP_COL]):
        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])

    # Stable sort keeps the original row order for equal timestamps.
    return df.sort_values(TIMESTAMP_COL, kind="mergesort").reset_index(drop=True)


def sensor_cols_for(df: pd.DataFrame) -> list[str]:
    """Return the required sensor columns plus any optional ones present in df."""
    return SENSOR_COLS + [c for c in OPTIONAL_SENSOR_COLS if c in df.columns]


def feature_names(sensor_cols: list[str] = SENSOR_COLS) -> list[str]:
    """Return the feature column names produced by build_features(), in order."""
    return [
        f"{col}_{suffix}"
        for col in sensor_cols
        for suffix in ("rmean", "rstd", "diff", "dev")
    ]


def time_of_day_slots(timestamps: pd.Series, slot_seconds: int = SLOT_SECONDS) -> np.ndarray:
    """Map timestamps (elapsed seconds or datetimes) to time-of-day slot numbers."""
    if pd.api.types.is_datetime64_any_dtype(timestamps):
        seconds = (timestamps - timestamps.dt.normalize()).dt.total_seconds()
    else:
        seconds = timestamps.astype(float) % SECONDS_PER_DAY
    return (seconds // slot_seconds).astype(int).to_numpy()


def expected_values(df: pd.DataFrame, profile: dict, sensor_cols: list[str]) -> pd.DataFrame:
    """Normal value of each sensor at each row's time of day."""
    slots = time_of_day_slots(df[TIMESTAMP_COL], profile["slot_seconds"])
    values = profile["expected"].loc[slots, sensor_cols].to_numpy()
    return pd.DataFrame(values, index=df.index, columns=sensor_cols)


def residuals(
    df: pd.DataFrame,
    profile: dict,
    sensor_cols: list[str],
    remove_common_mode: bool = True,
    scaled: bool = True,
) -> pd.DataFrame:
    """Reading minus the normal value for that time of day, per sensor.

    remove_common_mode subtracts, at each timestep, the median pressure residual
    across junctions: a shift shared by every junction (e.g. the tank running a bit
    fuller than usual) is not a local fault. scaled divides by the per-sensor
    residual std, giving z-scores.
    """
    resid = df[sensor_cols].astype(float) - expected_values(df, profile, sensor_cols)
    pressure = [c for c in sensor_cols if c.startswith("P_")]
    if remove_common_mode and pressure:
        resid[pressure] = resid[pressure].sub(resid[pressure].median(axis=1), axis=0)
    if scaled:
        resid = resid / pd.Series(profile["std"])[sensor_cols]
    return resid


def fit_profile(df: pd.DataFrame, sensor_cols: list[str], slot_seconds: int = SLOT_SECONDS) -> dict:
    """Learn the normal daily pattern: expected value per slot and residual std per sensor."""
    n_slots = SECONDS_PER_DAY // slot_seconds
    slots = time_of_day_slots(df[TIMESTAMP_COL], slot_seconds)
    expected = df[sensor_cols].astype(float).groupby(slots).mean().reindex(range(n_slots))
    # Fill any slot never seen in training from its neighbours (wrapping midnight).
    expected = pd.concat([expected] * 3, ignore_index=True).interpolate(limit_direction="both")
    expected = expected.iloc[n_slots:2 * n_slots].reset_index(drop=True)

    profile = {"slot_seconds": slot_seconds, "expected": expected}
    resid_std = residuals(df, profile, sensor_cols, scaled=False).std()
    std, floored = {}, []
    for col in sensor_cols:
        if col.startswith("P_"):
            floor = PRESSURE_STD_FLOOR
        else:
            floor = max(FLOW_STD_FLOOR_REL * float(df[col].abs().mean()), FLOW_STD_FLOOR_ABS)
        std[col] = max(float(resid_std[col]), floor)
        if std[col] == floor:
            floored.append(col)
    profile["std"] = std
    profile["floored"] = floored
    return profile


def build_features(
    df: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
    sensor_cols: list[str] | None = None,
    profile: dict | None = None,
) -> pd.DataFrame:
    """Compute rolling-window features for every sensor column.

    With a time-of-day profile, features are computed on residual z-scores
    (reading minus normal-for-this-time, see residuals()); without one, on raw values.
    Per sensor X this adds X_rmean (rolling mean), X_rstd (rolling std),
    X_diff (change since previous step) and X_dev (value minus rolling mean).
    The first window-1 warm-up rows are dropped. The original index and the
    timestamp column (if present) are kept. Only sensor columns are read, so
    scenario/leak_pipe labels can never leak into the features.
    sensor_cols defaults to sensor_cols_for(df); detect.py passes the trained list.
    """
    if sensor_cols is None:
        sensor_cols = sensor_cols_for(df)
    if profile is not None:
        sensors = residuals(df, profile, sensor_cols)
    else:
        sensors = df[sensor_cols].astype(float)
    rolling = sensors.rolling(window=window, min_periods=window)
    rmean = rolling.mean()
    rstd = rolling.std()
    diff = sensors.diff()
    dev = sensors - rmean

    parts = {}
    for col in sensor_cols:
        parts[f"{col}_rmean"] = rmean[col]
        parts[f"{col}_rstd"] = rstd[col]
        parts[f"{col}_diff"] = diff[col]
        parts[f"{col}_dev"] = dev[col]
    cols = feature_names(sensor_cols)
    feats = pd.DataFrame(parts, index=df.index)[cols]

    if TIMESTAMP_COL in df.columns:
        feats.insert(0, TIMESTAMP_COL, df[TIMESTAMP_COL])

    return feats.dropna(subset=cols)


def _features_by_segment(
    df: pd.DataFrame, window: int, sensor_cols: list[str], profile: dict
) -> pd.DataFrame:
    """Build features separately for each contiguous run of one scenario.

    This keeps rolling windows from straddling a scenario switch, so e.g. the
    first rows of a "normal" run never contain leftover leak readings.
    The scenario column is re-attached afterwards for filtering and reporting only.
    """
    segment_id = (df[SCENARIO_COL] != df[SCENARIO_COL].shift()).cumsum()
    pieces = []
    for _, segment in df.groupby(segment_id, sort=False):
        feats = build_features(segment, window, sensor_cols, profile)
        feats[SCENARIO_COL] = segment.loc[feats.index, SCENARIO_COL]
        pieces.append(feats)
    return pd.concat(pieces) if pieces else pd.DataFrame()


def train(
    csv_path: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    window: int = DEFAULT_WINDOW,
) -> dict:
    """Fit StandardScaler + IsolationForest on normal data and save the bundle.

    Returns:
        The saved bundle dict (scaler, model, sensor_cols, feature_cols, window,
        score_p01, score_p50, score_p99, baseline_means, baseline_stds, profile,
        sklearn_version).
    """
    df = load_data(csv_path)
    sensor_cols = sensor_cols_for(df)
    normal_raw = df[df[SCENARIO_COL] == NORMAL_SCENARIO]
    if len(normal_raw) == 0:
        raise ValueError(f"No '{NORMAL_SCENARIO}' rows in {csv_path}.")
    profile = fit_profile(normal_raw, sensor_cols)
    feats = _features_by_segment(df, window, sensor_cols, profile)
    cols = feature_names(sensor_cols)

    normal_feats = feats[feats[SCENARIO_COL] == NORMAL_SCENARIO] if len(feats) else feats
    if len(normal_feats) == 0:
        raise ValueError(
            f"No usable '{NORMAL_SCENARIO}' rows in {csv_path} after the "
            f"{window}-step warm-up. Need normal runs longer than the window."
        )

    X_normal = normal_feats[cols].to_numpy()
    scaler = StandardScaler().fit(X_normal)
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
    ).fit(scaler.transform(X_normal))

    # Higher decision_function = more normal. detect.py maps the median normal score
    # to severity 0 and p01 (the edge of normal) to 40, the start of yellow.
    scores = model.decision_function(scaler.transform(X_normal))
    score_p01, score_p50, score_p99 = (float(v) for v in np.percentile(scores, [1, 50, 99]))

    baseline_means = {c: float(v) for c, v in normal_raw[sensor_cols].mean().items()}
    baseline_stds = {c: float(v) for c, v in normal_raw[sensor_cols].std().items()}

    bundle = {
        "scaler": scaler,
        "model": model,
        "sensor_cols": sensor_cols,
        "feature_cols": cols,
        "window": window,
        "score_p01": score_p01,
        "score_p50": score_p50,
        "score_p99": score_p99,
        "baseline_means": baseline_means,
        "baseline_stds": baseline_stds,
        "profile": profile,
        "sklearn_version": sklearn.__version__,
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)

    # Sanity check: share of rows flagged anomalous per scenario.
    flagged = model.predict(scaler.transform(feats[cols].to_numpy())) == -1
    flag_rate = pd.Series(flagged, index=feats.index).groupby(feats[SCENARIO_COL]).mean()

    print("=== AquaWatch training summary ===")
    print(f"Data:              {csv_path}")
    print(f"Normal rows used:  {len(normal_feats)} (of {len(df)} total rows)")
    print(f"Features:          {len(cols)} ({len(sensor_cols)} sensors x 4, window={window})")
    print(f"Score p01/p50/p99: {score_p01:.4f} / {score_p50:.4f} / {score_p99:.4f}")
    print(f"Residual std at sensor-accuracy floor: {len(profile['floored'])}/{len(sensor_cols)} sensors")
    print(f"Saved model to:    {model_path}")
    print("Share flagged anomalous per scenario:")
    for scenario, rate in flag_rate.items():
        print(f"  {scenario:<14} {rate:6.1%}")

    return bundle


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Train the AquaWatch anomaly detector.")
    parser.add_argument("--data", required=True, help="Path to the simulator sensor CSV.")
    parser.add_argument("--out", default=DEFAULT_MODEL_PATH, help="Where to save the model bundle.")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="Rolling window size in timesteps.")
    args = parser.parse_args(argv)
    train(args.data, args.out, args.window)


if __name__ == "__main__":
    main()
