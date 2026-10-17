"""Real-time leak detection and pipe localization for AquaWatch.

Usage:
    python -m ai.detect --data path/to/scenario.csv [--model ai/model.joblib] [--inp data/Net1.inp]

LeakDetector.detect() takes the most recent raw sensor rows and returns one alert
dict for the latest timestep: how abnormal it is (severity), which pipe is the
likely culprit, what kind of event it looks like, and a plain-English explanation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .topology import DEFAULT_INP_PATH, load_network, pipe_ends
from .train import (
    DEFAULT_MODEL_PATH,
    LEAK_PIPE_COL,
    SCENARIO_COL,
    TIMESTAMP_COL,
    build_features,
    load_data,
)

DEFAULT_CONSECUTIVE = 3

# --- Severity bands (0-100): <40 green, 40-70 yellow, >70 red ---
YELLOW_AT = 40
RED_ABOVE = 70

# --- Localization weights ---
PRESSURE_WEIGHT = 1.0  # relative pressure drop at a pipe's end junctions
FLOW_WEIGHT = 1.0      # extra water "disappearing" at a junction (mass-balance residual)
SMOOTHING = 0.3        # share of a pipe's score taken from its neighbouring pipes

# --- Event classification thresholds (fractions of baseline; tune on real data) ---
NETWORK_DROP_MIN = 0.02    # a junction counts as "dropped" if pressure fell >= 2%
NETWORK_DROP_SHARE = 0.7   # demand spike: >= 70% of junctions dropped...
UNIFORM_CV_MAX = 0.5       # ...by similar amounts (std / mean of drops <= 0.5)
DOMINANCE_RATIO = 2.5      # top pipe is "dominant" if its score >= 2.5x an even share
RUPTURE_STEP_DROP = 0.05   # one-step pressure fall >= 5% of baseline = sudden (rupture)

EPS = 1e-9


def _sensor_id(col: str) -> str:
    """'P_22' -> '22', 'F_112' -> '112'."""
    return col.split("_", 1)[1]


def _sensors_from_features(feature_cols: list[str]) -> list[str]:
    """Recover the sensor list from feature names (for bundles without sensor_cols)."""
    return list(dict.fromkeys(f.rsplit("_", 1)[0] for f in feature_cols))


def _to_builtin(value: Any) -> Any:
    """Make timestamps and numpy scalars JSON-friendly."""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


class LeakDetector:
    """Scores sensor windows with the trained model and localizes the likely pipe."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        inp_path: str | Path = DEFAULT_INP_PATH,
        consecutive: int = DEFAULT_CONSECUTIVE,
    ) -> None:
        bundle = joblib.load(model_path)
        self.scaler = bundle["scaler"]
        self.model = bundle["model"]
        self.feature_cols: list[str] = bundle["feature_cols"]
        self.window: int = int(bundle["window"])
        self.score_p01: float = bundle["score_p01"]
        self.score_p99: float = bundle["score_p99"]
        self.baseline_means: dict[str, float] = bundle["baseline_means"]
        self.baseline_stds: dict[str, float] = bundle.get("baseline_stds", {})
        self.sensor_cols: list[str] = bundle.get("sensor_cols") or _sensors_from_features(self.feature_cols)
        self.consecutive = consecutive

        self.graph = load_network(inp_path)
        all_ends = pipe_ends(self.graph)
        self.pipe_ends = pipe_ends(self.graph, include_pumps=False)
        self._link_kind = {d["pipe_id"]: d["kind"] for _, _, d in self.graph.edges(data=True)}

        # Sensor lookup: junction id -> pressure column, link id -> flow column.
        self.pressure_cols = {_sensor_id(c): c for c in self.sensor_cols if c.startswith("P_")}
        self.flow_cols = {_sensor_id(c): c for c in self.sensor_cols if c.startswith("F_")}
        unknown = [c for n, c in self.pressure_cols.items() if n not in self.graph]
        unknown += [c for p, c in self.flow_cols.items() if p not in all_ends]
        if unknown:
            raise ValueError(f"Sensor columns {unknown} don't match any node/link in {inp_path}")

        # Pipes that share a node, for neighbour smoothing.
        self._pipe_neighbors = {
            p: [q for q, eq in self.pipe_ends.items() if q != p and set(ep) & set(eq)]
            for p, ep in self.pipe_ends.items()
        }

        # Junctions where every connected link has a flow sensor, so inflow - outflow
        # can be computed. Each entry: [(flow column, +1 if flowing in / -1 if out)].
        self._balance_terms: dict[str, list[tuple[str, int]]] = {}
        for node, data in self.graph.nodes(data=True):
            if data["type"] != "junction":
                continue
            terms = []
            for link, (start, end) in all_ends.items():
                if node in (start, end):
                    terms.append((self.flow_cols.get(link), 1 if end == node else -1))
            if terms and all(col is not None for col, _ in terms):
                self._balance_terms[node] = terms

        self.reset()

    def reset(self) -> None:
        """Forget the consecutive-anomaly streak (call between independent replays)."""
        self._streak = 0

    # ------------------------------------------------------------------ detect

    def detect(self, window_df: pd.DataFrame) -> dict[str, Any]:
        """Score the latest row of window_df and return the alert dict.

        window_df holds recent raw sensor rows, oldest first; it needs at least
        `window` rows (fewer returns a green "warming up" result). Call once per
        new timestep: the consecutive-anomaly streak is kept between calls.
        """
        recent = window_df.tail(2 * self.window - 1)
        timestamp = _to_builtin(
            recent[TIMESTAMP_COL].iloc[-1] if TIMESTAMP_COL in recent else recent.index[-1]
        )
        if len(recent) < self.window:
            return self._result(
                timestamp, False, 0, None, self._uniform_scores(), "normal",
                f"Warming up: need {self.window} readings, have {len(recent)}.",
            )

        feats = build_features(recent, self.window, self.sensor_cols)
        X = feats[self.feature_cols].iloc[[-1]].to_numpy(dtype=float)
        score = float(self.model.decision_function(self.scaler.transform(X))[0])
        severity = self._severity(score)

        self._streak = self._streak + 1 if severity >= YELLOW_AT else 0
        is_anomaly = self._streak >= self.consecutive

        row = recent.iloc[-1][self.sensor_cols].astype(float)
        pipe_scores = self.localize(row)
        event_type = self.classify_event(feats, pipe_scores) if is_anomaly else "normal"

        # Unconfirmed blips and demand spikes never go red.
        if not is_anomaly or event_type == "demand_spike":
            severity = min(severity, RED_ABOVE)

        suspected = max(pipe_scores, key=pipe_scores.get) if event_type in ("leak", "rupture") else None
        explanation = self.explain(row, event_type, suspected, pipe_scores, severity)
        return self._result(timestamp, is_anomaly, severity, suspected, pipe_scores, event_type, explanation)

    def _severity(self, score: float) -> int:
        """Map decision_function linearly: score_p99 -> 0, score_p01 -> 100, clipped."""
        span = self.score_p99 - self.score_p01
        if span <= EPS:
            return 100 if score < self.score_p01 else 0
        return int(round(np.clip((self.score_p99 - score) / span * 100, 0, 100)))

    @staticmethod
    def _level(severity: int) -> str:
        if severity > RED_ABOVE:
            return "red"
        return "yellow" if severity >= YELLOW_AT else "green"

    def _uniform_scores(self) -> dict[str, float]:
        return {p: 1.0 / len(self.pipe_ends) for p in self.pipe_ends}

    def _result(self, timestamp: Any, is_anomaly: bool, severity: int, suspected: str | None,
                pipe_scores: dict[str, float], event_type: str, explanation: str) -> dict[str, Any]:
        return {
            "timestamp": timestamp,
            "is_anomaly": bool(is_anomaly),
            "severity": int(severity),
            "level": self._level(severity),
            "suspected_pipe": suspected,
            "pipe_scores": pipe_scores,
            "event_type": event_type,
            "explanation": explanation,
        }

    # ---------------------------------------------------------------- localize

    def _pressure_drops(self, values: pd.Series) -> dict[str, float]:
        """Relative pressure drop vs baseline per sensored junction (0.1 = fell 10%)."""
        return {
            node: (self.baseline_means[col] - values[col]) / max(abs(self.baseline_means[col]), EPS)
            for node, col in self.pressure_cols.items()
        }

    def localize(self, row: pd.Series) -> dict[str, float]:
        """Score each pipe by how likely it is to be leaking; scores sum to 1.

        1. Pressure: each junction's relative drop vs baseline, minus the median drop
           across junctions (removes network-wide shifts such as daily demand), so only
           the local "hole" stands out. A pipe gets the average of its sensored ends;
           pipes touching tank 2 / reservoir 9 just use their one sensored end.
        2. Flow balance: at each fully metered junction, inflow - outflow should equal
           its normal consumption. Any extra water missing is added to the pipes
           touching that junction.
        3. Smoothing: blend in a little of the neighbouring pipes' scores.
        """
        drops = self._pressure_drops(row)
        common = float(np.median(list(drops.values()))) if drops else 0.0
        local = {node: d - common for node, d in drops.items()}

        raw: dict[str, float] = {}
        for pipe, ends in self.pipe_ends.items():
            end_drops = [local[n] for n in ends if n in local]
            raw[pipe] = PRESSURE_WEIGHT * float(np.mean(end_drops)) if end_drops else 0.0

        for node, terms in self._balance_terms.items():
            now = sum(sign * row[col] for col, sign in terms)
            base = sum(sign * self.baseline_means[col] for col, sign in terms)
            scale = sum(abs(self.baseline_means[col]) for col, _ in terms) + EPS
            missing = max((now - base) / scale, 0.0)
            for pipe, ends in self.pipe_ends.items():
                if node in ends:
                    raw[pipe] += FLOW_WEIGHT * missing

        smoothed = {
            p: (1 - SMOOTHING) * s
            + SMOOTHING * (float(np.mean([raw[q] for q in self._pipe_neighbors[p]])) if self._pipe_neighbors[p] else s)
            for p, s in raw.items()
        }
        clipped = {p: max(s, 0.0) for p, s in smoothed.items()}
        total = sum(clipped.values())
        if total <= EPS:
            return self._uniform_scores()
        return {p: s / total for p, s in clipped.items()}

    # ---------------------------------------------------------------- classify

    def classify_event(self, window_feats: pd.DataFrame, pipe_scores: dict[str, float]) -> str:
        """Label a confirmed anomaly as "demand_spike", "rupture" or "leak".

        - demand_spike: most junctions dropped by similar amounts and no pipe stands out.
        - rupture: some junction's pressure fell sharply in a single step.
        - leak: everything else (a smaller, gradual, localized drift).
        """
        last = window_feats.iloc[-1]
        current = pd.Series({c: last[f"{c}_rmean"] + last[f"{c}_dev"] for c in self.sensor_cols})
        drops = np.array(list(self._pressure_drops(current).values()))

        mean_drop = float(drops.mean()) if drops.size else 0.0
        uniform = (
            drops.size > 0
            and float((drops >= NETWORK_DROP_MIN).mean()) >= NETWORK_DROP_SHARE
            and mean_drop > EPS
            and float(drops.std()) / mean_drop <= UNIFORM_CV_MAX
        )
        dominant = max(pipe_scores.values()) >= DOMINANCE_RATIO / len(pipe_scores)
        if uniform and not dominant:
            return "demand_spike"

        biggest_step = max(
            (-window_feats[f"{col}_diff"].min() / max(abs(self.baseline_means[col]), EPS)
             for col in self.pressure_cols.values()),
            default=0.0,
        )
        return "rupture" if biggest_step >= RUPTURE_STEP_DROP else "leak"

    # ----------------------------------------------------------------- explain

    def _top_changes(self, row: pd.Series, n: int = 3) -> str:
        """Describe the n sensors furthest from baseline (by z-score) with % change."""
        ranked = []
        for col in self.sensor_cols:
            base = self.baseline_means[col]
            std = self.baseline_stds.get(col) or max(abs(base) * 0.01, EPS)
            ranked.append((abs(row[col] - base) / std, col, (row[col] - base) / max(abs(base), EPS)))
        ranked.sort(reverse=True)

        phrases = []
        for _, col, change in ranked[:n]:
            sensor = _sensor_id(col)
            if col.startswith("P_"):
                what = f"pressure at {sensor}"
            else:
                what = f"flow {'through pump' if self._link_kind[sensor] == 'pump' else 'in pipe'} {sensor}"
            phrases.append(f"{what} {'fell' if change < 0 else 'rose'} {abs(change):.0%}")
        text = ", ".join(phrases)
        return text[:1].upper() + text[1:] + " compared with normal levels."

    def explain(self, row: pd.Series, event_type: str, suspected_pipe: str | None,
                pipe_scores: dict[str, float], severity: int) -> str:
        """Build a short, template-based explanation of the current result."""
        if event_type == "normal":
            if self._streak == 0:
                return "All sensors within normal range."
            return (
                f"Unusual readings for {self._streak} step(s) (severity {severity}); waiting for "
                f"{self.consecutive} in a row before alerting. {self._top_changes(row)}"
            )

        changes = self._top_changes(row)
        if event_type == "demand_spike":
            return (
                f"Likely demand spike across the network (severity {severity}). {changes} "
                "Pressure fell at most junctions with no single hotspot, so this is probably "
                "higher demand, not a leak."
            )

        start, end = self.pipe_ends[suspected_pipe]
        confidence = pipe_scores[suspected_pipe]
        reason = (
            "Sudden, large localized pressure drop, consistent with a pipe burst."
            if event_type == "rupture"
            else "Gradual, localized pattern, unlikely to be a demand spike."
        )
        return (
            f"Likely {event_type} on pipe {suspected_pipe} ({start}→{end}), "
            f"confidence {confidence:.0%}. {changes} {reason}"
        )


def replay(
    csv_path: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    inp_path: str | Path = DEFAULT_INP_PATH,
    consecutive: int = DEFAULT_CONSECUTIVE,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Stream a scenario CSV row by row through detect(), printing alert changes.

    Prints when an alert starts, changes (event type / pipe / level) or clears,
    with the CSV's ground truth alongside when present. Returns every result.
    """
    df = load_data(csv_path)
    detector = LeakDetector(model_path, inp_path, consecutive)
    history = 2 * detector.window - 1

    results: list[dict[str, Any]] = []
    prev_key: tuple | None = None
    for i in range(len(df)):
        res = detector.detect(df.iloc[max(0, i - history + 1): i + 1])
        results.append(res)

        key = (res["is_anomaly"], res["event_type"], res["suspected_pipe"], res["level"])
        was_alerting = prev_key is not None and prev_key[0]
        if verbose and key != prev_key and (res["is_anomaly"] or was_alerting):
            truth = ""
            if SCENARIO_COL in df:
                leak = df[LEAK_PIPE_COL].iloc[i] if LEAK_PIPE_COL in df else None
                truth = f"  [truth: {df[SCENARIO_COL].iloc[i]}, pipe {leak if pd.notna(leak) else '-'}]"
            if res["is_anomaly"]:
                print(f"[{res['timestamp']}] {res['level'].upper():<6} sev={res['severity']:>3} "
                      f"{res['event_type']}: {res['explanation']}{truth}")
            else:
                print(f"[{res['timestamp']}] cleared{truth}")
        prev_key = key

    if verbose:
        alerts = sum(r["is_anomaly"] for r in results)
        print(f"=== Replayed {len(results)} rows, {alerts} alerting steps ===")
    return results


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Replay a sensor CSV through the AquaWatch detector.")
    parser.add_argument("--data", required=True, help="Path to the scenario sensor CSV.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Trained model bundle from ai.train.")
    parser.add_argument("--inp", default=DEFAULT_INP_PATH, help="EPANET network file.")
    parser.add_argument("--consecutive", type=int, default=DEFAULT_CONSECUTIVE,
                        help="Anomalous steps in a row required before alerting.")
    args = parser.parse_args(argv)
    replay(args.data, args.model, args.inp, args.consecutive)


if __name__ == "__main__":
    main()
