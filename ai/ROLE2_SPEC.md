# Role 2 — AI Detection & Pipe Localization (spec)

Stack: Python, pandas, NumPy, scikit-learn, joblib, NetworkX. Target: IBM LinuxONE (s390x) — no x86-only binaries.

## Data contract (from Role 1, simulator/)
One row per timestep:
- `timestamp`
- `P_J1..P_J8` junction pressures
- `F_P1..F_P10` pipe flows
- `scenario` in {normal, leak, rupture, demand_spike}
- `leak_pipe` (ground truth, evaluation only)
Graph: `simulator/network.py: build_network()` -> NetworkX graph, edges carry `pipe_id`.
Until it exists, use a small fake generator for testing.

## ai/__init__.py
`from .detect import LeakDetector`

## ai/train.py
1. `build_features(df, window=10)` — shared with detect.py. Per sensor: rolling mean, rolling std, diff(), deviation from rolling mean; optional time-of-day. Drop warm-up NaNs.
2. `train(csv_path, model_path="ai/model.joblib")`:
   - use only `scenario == "normal"` rows
   - StandardScaler + IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
   - store normal-data decision_function percentiles p01/p99 for 0–100 scaling
   - joblib bundle: scaler, model, feature_cols, window, score_p01, score_p99, baseline_means (per-sensor normal mean)
3. CLI: `python -m ai.train --data ... --out ...`

## ai/detect.py
`class LeakDetector(model_path, graph)` with `detect(window_df) -> dict`:
```
{timestamp, is_anomaly, severity (0-100), level (green/yellow/red),
 suspected_pipe, pipe_scores {pipe: prob}, event_type (leak/rupture/demand_spike),
 explanation}
```
- Severity: map decision_function linearly p99->0, p01->100, clip. Bands: <40 green, 40–70 yellow, >70 red. Require 2–3 consecutive anomalous steps before alerting.
- `localize()`: per-junction pressure drop vs baseline_means; pipe score from its two end junctions + flow-balance residual (inflow − outflow ≈ 0 at each junction); optional neighbor smoothing via nx.neighbors; normalize to sum 1; argmax = suspected_pipe.
- `classify_event()`: rupture = large sudden localized drop; leak = gradual localized drift; demand_spike = network-wide drop, no local imbalance (do NOT flag red).
- `explain()`: template text naming top 2–3 sensors by z-score, % change, time window, suspected pipe + confidence.
- `__main__`: replay a scenario CSV row by row and print alerts.

## Handoffs
- Role 3: detect() output + suspected_pipe for recall / detection time / false alarms / localization accuracy vs threshold baseline.
- Role 4: pipe_scores, level, explanation for map colors + alert box.
