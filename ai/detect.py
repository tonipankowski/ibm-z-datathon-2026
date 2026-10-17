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
    residuals,
)

DEFAULT_CONSECUTIVE = 3

# --- Severity bands (0-100): <40 green, 40-70 yellow, >70 red ---
YELLOW_AT = 40
RED_ABOVE = 70

# --- Residual check (runs next to the IsolationForest, which saturates on large
# out-of-range values). Largest |z| over junction mass balances and local junction
# pressures; Z_ALERT sigma maps to YELLOW_AT. 4 sigma rather than the classic 3 because
# ~18 checks run every step (at 3 sigma one would trip by chance ~5% of steps).
Z_ALERT = 4.0

# --- Localization weights (both terms are in sigmas, so directly comparable) ---
PRESSURE_WEIGHT = 1.0  # local pressure drop at a pipe's end junctions
FLOW_WEIGHT = 1.0      # extra water "disappearing" at a junction (mass-balance residual)
SMOOTHING = 0.3        # share of a pipe's score taken from its neighbouring pipes
# Flow meters sit at each pipe's start node (.inp node1), as in the simulator, where
# F_12 measures the upstream half of pipe 12. A junction's balance then encloses only
# the pipes that END there, so its missing water is blamed on those pipes alone.
# Set False if meter positions are unknown: the imbalance is shared by all touching pipes.
METER_AT_START = True

# --- Event classification thresholds (in residual sigmas) ---
# Demand spike: missing water spread over many junctions, pressures falling everywhere.
DEMAND_MIN_JUNCTIONS = 3   # >= 3 junction balances at >= Z_ALERT (a pipe leak touches <= 2)
DEMAND_MAX_SHARE = 0.5     # ...and no single junction holds more than half the missing water
NETWORK_DROP_SHARE = 0.7   # ...and >= 70% of junction pressures fell...
PRESSURE_FALL_Z = 1.0      # ...by more than 1 sigma since just before the alert
# Rupture: sudden local pressure fall (beyond the network-wide change) in one step.
# 10 sigma = 1 m; normal data never exceeds ~1.3 sigma, since pump switching is network-wide.
RUPTURE_STEP_Z = 10.0
# Rupture also if the estimated lost flow is large. Orifice law Q = Cd*A*sqrt(2gh) with
# Cd = 0.62 and h ~ 80 m (Net1 sensor pressures are ~72-88 m): a 5 mm hole loses ~0.48 L/s,
# a 50 mm hole ~48 L/s. 5 L/s is the midpoint on a log scale, 10x from each (a ~16 mm hole);
# the +-10% head range moves those flows by only ~5%. Large mains near the tank barely lose
# pressure even in a burst, so lost flow is the more reliable size signal.
RUPTURE_FLOW_LPS = 5.0

# --- Ambiguous localization ---
# Pipes in the same junction's balance get identical mass-balance evidence; only the
# pressure term separates them. If the runner-up scores >= 80% of the top pipe, that
# separating evidence is under ~20% of the score, too small to trust against ~1 sigma
# pressure noise, so the explanation names both pipes.
AMBIGUOUS_RATIO = 0.8

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
    """Scores sensor windows and localizes the likely pipe.

    Two detectors run side by side on time-of-day residuals: the IsolationForest
    (unusual combinations of readings) and a residual check (largest junction
    mass-balance / local pressure z-score), which keeps growing with event size
    where the forest saturates. Severity is the larger of the two.
    """

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
        self.score_p50: float | None = bundle.get("score_p50")
        self.sensor_cols: list[str] = bundle.get("sensor_cols") or _sensors_from_features(self.feature_cols)
        self.profile: dict | None = bundle.get("profile")
        if self.profile is None:
            raise ValueError(f"{model_path} has no time-of-day profile; retrain it with ai.train.")
        self.resid_std: dict[str, float] = self.profile["std"]
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

        # Pipes whose unmetered length lies inside each junction's balance.
        self._balance_pipes = {
            node: [p for p, (start, end) in self.pipe_ends.items()
                   if end == node or (not METER_AT_START and start == node)]
            for node in self.graph.nodes
        }

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
        # Std of each balance: independent meter errors add in quadrature.
        self._balance_sd = {
            node: float(np.sqrt(sum(self.resid_std[col] ** 2 for col, _ in terms)))
            for node, terms in self._balance_terms.items()
        }

        self.reset()

    def reset(self) -> None:
        """Forget the anomaly streak and current event (call between independent replays)."""
        self._streak = 0
        self._event_type: str | None = None
        self.lost_flow_lps = 0.0  # latest estimate of water unaccounted for, in L/s

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
            self.lost_flow_lps = 0.0
            return self._result(
                timestamp, False, 0, None, self._uniform_scores(), "normal",
                f"Warming up: need {self.window} readings, have {len(recent)}.",
            )

        feats = build_features(recent, self.window, self.sensor_cols, self.profile)
        X = feats[self.feature_cols].iloc[[-1]].to_numpy(dtype=float)
        score = float(self.model.decision_function(self.scaler.transform(X))[0])

        # Residuals vs the normal value for this time of day, in sensor units:
        # `tod` keeps network-wide shifts, `local` removes them (see train.residuals).
        tod = residuals(recent, self.profile, self.sensor_cols, remove_common_mode=False, scaled=False)
        local = residuals(recent, self.profile, self.sensor_cols, scaled=False)
        row = local.iloc[-1]

        severity = max(self._severity(score), self._residual_severity(row))
        self.lost_flow_lps = self.lost_flow(row)
        self._streak = self._streak + 1 if severity >= YELLOW_AT else 0
        is_anomaly = self._streak >= self.consecutive

        pipe_scores = self.localize(row)
        # The event type is decided when the alert is confirmed and kept until it clears
        # (the onset step scrolls out of the window); a leak may still escalate to rupture.
        if is_anomaly:
            current = self.classify_event(tod, local, pipe_scores)
            if self._event_type is None or (self._event_type == "leak" and current == "rupture"):
                self._event_type = current
            event_type = self._event_type
        else:
            self._event_type = None
            event_type = "normal"

        # Unconfirmed blips and demand spikes never go red.
        if not is_anomaly or event_type == "demand_spike":
            severity = min(severity, RED_ABOVE)

        suspected = max(pipe_scores, key=pipe_scores.get) if event_type in ("leak", "rupture") else None
        explanation = self.explain(row, event_type, suspected, pipe_scores, severity)
        return self._result(timestamp, is_anomaly, severity, suspected, pipe_scores, event_type, explanation)

    def _severity(self, score: float) -> int:
        """Map decision_function linearly to 0-100, clipped.

        The median normal score -> 0 and score_p01 (the edge of normal) -> YELLOW_AT,
        so 99% of normal data stays green; further below p01 climbs towards red.
        Old bundles without score_p50 fall back to p99 -> 0, p01 -> 100.
        """
        if self.score_p50 is None:
            top, at = self.score_p99, 100.0
        else:
            top, at = self.score_p50, float(YELLOW_AT)
        span = top - self.score_p01
        if span <= EPS:
            return 100 if score < self.score_p01 else 0
        return int(round(np.clip((top - score) / span * at, 0, 100)))

    def balance_z(self, row: pd.Series) -> dict[str, float]:
        """Per metered junction: (inflow - outflow) minus its normal value, in sigmas.

        Positive = more water leaving the junction's metered boundary than usual
        (extra consumption or a leak). Flow residuals cancel network-wide shifts,
        because conservation ties each balance to that junction's demand alone.
        """
        return {
            node: sum(sign * row[col] for col, sign in terms) / self._balance_sd[node]
            for node, terms in self._balance_terms.items()
        }

    def lost_flow(self, row: pd.Series) -> float:
        """Estimated water unaccounted for, in L/s.

        Sum of the extra outflow (m3/s above normal for this time of day) over the
        junctions whose balance is significantly high (>= Z_ALERT), converted to L/s.
        For a leak this is the leak flow; for a demand spike, the extra consumption.
        """
        total = 0.0
        for node, z in self.balance_z(row).items():
            if z >= Z_ALERT:
                total += max(sum(sign * row[col] for col, sign in self._balance_terms[node]), 0.0)
        return total * 1000.0

    def pressure_z(self, row: pd.Series) -> dict[str, float]:
        """Per pressure junction: local residual in sigmas (negative = lower than normal)."""
        return {node: row[col] / self.resid_std[col] for node, col in self.pressure_cols.items()}

    def _residual_severity(self, row: pd.Series) -> int:
        """Severity from the largest |z|: 0 sigma -> 0, Z_ALERT -> YELLOW_AT, linear, clipped."""
        zs = [*self.balance_z(row).values(), *self.pressure_z(row).values()]
        z_max = max((abs(z) for z in zs), default=0.0)
        return int(round(np.clip(z_max / Z_ALERT * YELLOW_AT, 0, 100)))

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

    def localize(self, row: pd.Series) -> dict[str, float]:
        """Score each pipe by how likely it is to be leaking; scores sum to 1.

        row holds residuals (reading minus normal-for-this-time, network-wide pressure
        shift removed). Both evidence terms are in sigmas, never percentages: flows can
        sit near zero or reverse, where a % change is meaningless.
        1. Pressure: each junction's local drop in sigmas. A pipe gets the average of
           its sensored ends; pipes touching tank 2 / reservoir 9 use their one end.
        2. Flow balance: at each fully metered junction, inflow - outflow should equal
           its normal consumption for this time. Extra water missing (in sigmas) is
           added to the pipes whose unmetered length lies inside that balance: with
           meters at pipe starts, the pipes ending at the junction. E.g. a leak midway
           along pipe 12 (12->13) raises F_12 but not F_113, so it shows up at
           junction 13 and is blamed on pipe 12, not on 113.
        3. Smoothing: blend in a little of the neighbouring pipes' scores.
        """
        drops = {node: -z for node, z in self.pressure_z(row).items()}

        raw: dict[str, float] = {}
        for pipe, ends in self.pipe_ends.items():
            end_drops = [drops[n] for n in ends if n in drops]
            raw[pipe] = PRESSURE_WEIGHT * float(np.mean(end_drops)) if end_drops else 0.0

        for node, z in self.balance_z(row).items():
            for pipe in self._balance_pipes[node]:
                raw[pipe] += FLOW_WEIGHT * max(z, 0.0)

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

    def classify_event(self, tod: pd.DataFrame, local: pd.DataFrame, pipe_scores: dict[str, float]) -> str:
        """Label a confirmed anomaly as "demand_spike", "rupture" or "leak".

        tod / local: recent residual windows with / without network-wide pressure shifts.
        All thresholds are in residual sigmas.
        - demand_spike: extra water is missing at many junctions with no single hotspot
          (so no pipe dominates), and most junction pressures fell since before the alert.
        - rupture: lost flow >= RUPTURE_FLOW_LPS, or some junction's local pressure
          fell >= RUPTURE_STEP_Z in one step.
        - leak: everything else (a smaller, localized loss).
        pipe_scores is accepted for API compatibility; the junction share test is its
        physical equivalent (a single leaking pipe concentrates the missing water).
        """
        missing = np.array([max(z, 0.0) for z in self.balance_z(local.iloc[-1]).values()])
        hot = int((missing >= Z_ALERT).sum())
        top_share = float(missing.max() / missing.sum()) if missing.sum() > EPS else 1.0

        pressure = list(self.pressure_cols.values())
        sd = pd.Series(self.resid_std)[pressure]
        fell = float((((tod[pressure].iloc[-1] - tod[pressure].iloc[0]) / sd) <= -PRESSURE_FALL_Z).mean())

        if hot >= DEMAND_MIN_JUNCTIONS and top_share <= DEMAND_MAX_SHARE and fell >= NETWORK_DROP_SHARE:
            return "demand_spike"

        if self.lost_flow(local.iloc[-1]) >= RUPTURE_FLOW_LPS:
            return "rupture"
        local_fall = float((-local[pressure].diff() / sd).max().max())
        return "rupture" if local_fall >= RUPTURE_STEP_Z else "leak"

    # ----------------------------------------------------------------- explain

    def _top_changes(self, row: pd.Series, n: int = 3) -> str:
        """Describe the n signals furthest from normal for this time of day, in sigmas.

        Candidates: each flow, each junction's local pressure, and each junction's
        mass balance (water unaccounted for). Sigmas, not %, because flows can be
        near zero or reverse direction.
        """
        ranked: list[tuple[float, str]] = []
        for node, z in self.pressure_z(row).items():
            metres = row[self.pressure_cols[node]]
            ranked.append((abs(z), f"pressure at {node} is {abs(metres):.2f} m ({abs(z):.1f}σ) "
                                   f"{'below' if z < 0 else 'above'} normal"))
        for link, col in self.flow_cols.items():
            z = row[col] / self.resid_std[col]
            what = "flow through pump" if self._link_kind[link] == "pump" else "flow in pipe"
            ranked.append((abs(z), f"{what} {link} is {abs(z):.1f}σ {'above' if z > 0 else 'below'} normal"))
        for node, z in self.balance_z(row).items():
            ranked.append((abs(z), f"water unaccounted for at junction {node} is {abs(z):.1f}σ "
                                   f"{'above' if z > 0 else 'below'} normal"))
        ranked.sort(key=lambda t: t[0], reverse=True)

        text = ", ".join(phrase for _, phrase in ranked[:n])
        return text[:1].upper() + text[1:] + " for this time of day."

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
        flow = self.lost_flow_lps
        if event_type == "demand_spike":
            extra = f" About {flow:.1f} L/s more water used than normal." if flow > 0 else ""
            return (
                f"Likely demand spike across the network (severity {severity}).{extra} {changes} "
                "Pressure fell at most junctions with no single hotspot, so this is probably "
                "higher demand, not a leak."
            )

        start, end = self.pipe_ends[suspected_pipe]
        confidence = pipe_scores[suspected_pipe]
        where = f"pipe {suspected_pipe} ({start}→{end}), confidence {confidence:.0%}."
        partner = self._ambiguous_partner(pipe_scores)
        if partner is not None:
            other, node = partner
            o_start, o_end = self.pipe_ends[other]
            where = (
                f"pipe {suspected_pipe} ({start}→{end}) or {other} ({o_start}→{o_end}), "
                f"confidence {confidence:.0%} / {pipe_scores[other]:.0%} (both feed junction {node}; "
                "an extra flow meter would separate them)."
            )
        lost = f" About {flow:.1f} L/s unaccounted for." if flow > 0 else ""
        if event_type == "rupture":
            reason = (
                f"Loss above {RUPTURE_FLOW_LPS:g} L/s at one location, consistent with a pipe burst."
                if flow >= RUPTURE_FLOW_LPS
                else "Sudden, large localized pressure drop, consistent with a pipe burst."
            )
        else:
            reason = "Small, localized loss, unlikely to be a demand spike."
        return f"Likely {event_type} on {where}{lost} {changes} {reason}"

    def _ambiguous_partner(self, pipe_scores: dict[str, float]) -> tuple[str, str] | None:
        """(runner-up pipe, shared junction) if the top two pipes can't be told apart.

        That is when both lie in the same junction's mass balance and the runner-up
        scores at least AMBIGUOUS_RATIO of the top pipe.
        """
        top, second = sorted(pipe_scores, key=pipe_scores.get, reverse=True)[:2]
        if pipe_scores[second] < AMBIGUOUS_RATIO * pipe_scores[top]:
            return None
        for node, pipes in self._balance_pipes.items():
            if top in pipes and second in pipes:
                return second, node
        return None


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
