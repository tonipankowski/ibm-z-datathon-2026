"""Quick developer check of the detector on Role 1's simulated scenarios.

Not the official evaluation (Role 3 owns that). Replays each CSV through
LeakDetector and scores the results against the *.truth.csv event windows.
Truth files and the scenario/leak_pipe columns are used here only for scoring.

Suites:
    generated     data/generated: leak / rupture on pipe 12, demand spike (one model)
    sensor_study  data/sensor_study: leaks / ruptures on pipes 11, 12, 31, 112 and a
                  demand spike, clean and noisy. Each variant gets its own model,
                  trained only on its own normal/train.<variant>.csv.

Usage:
    python -m ai.evaluate_quick [--suite generated|sensor_study|all] [--retrain] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .detect import LeakDetector
from .topology import DEFAULT_INP_PATH
from .train import DEFAULT_MODEL_PATH, TIMESTAMP_COL, load_data, train

GENERATED_DIR = Path("data/generated")
SENSOR_STUDY_DIR = Path("data/sensor_study")
DEFAULT_MODEL_DIR = Path("ai/models")


def generated_suite(model_path: Path) -> dict[str, Any]:
    """Cases for data/generated (all events on pipe 12)."""
    return {
        "name": "generated",
        "model": model_path,
        "train": GENERATED_DIR / "normal_train.csv",
        "holdout": GENERATED_DIR / "normal_holdout.csv",
        "cases": [
            {"name": f"{kind}" + (" (pipe 12)" if kind != "demand_spike" else ""),
             "csv": GENERATED_DIR / f"{kind}.csv", "truth": GENERATED_DIR / f"{kind}.truth.csv",
             "type": kind, "pipe": None if kind == "demand_spike" else "12"}
            for kind in ("leak", "rupture", "demand_spike")
        ],
    }


def sensor_study_suite(variant: str, model_dir: Path) -> dict[str, Any]:
    """Cases for data/sensor_study/<case>/ in the clean or noisy variant."""
    cases = []
    for folder in sorted(p for p in SENSOR_STUDY_DIR.iterdir() if p.is_dir() and p.name != "normal"):
        kind, _, pipe = folder.name.partition("_pipe")
        suffix = ".csv" if variant == "clean" else ".noisy.csv"
        cases.append({
            "name": folder.name, "csv": folder / f"{kind}{suffix}", "truth": folder / f"{kind}.truth.csv",
            "type": kind, "pipe": pipe or None,
        })
    return {
        "name": f"sensor_study/{variant}",
        "model": model_dir / f"sensor_study_{variant}.joblib",
        "train": SENSOR_STUDY_DIR / "normal" / f"train.{variant}.csv",
        "holdout": SENSOR_STUDY_DIR / "normal" / f"holdout.{variant}.csv",
        "cases": cases,
    }


def run(csv_path: Path, model_path: Path, inp_path: Path) -> pd.DataFrame:
    """Stream a CSV through a fresh detector; one row of results per timestep."""
    df = load_data(csv_path)
    detector = LeakDetector(model_path, inp_path)
    history = 2 * detector.window - 1
    rows = []
    for i in range(len(df)):
        res = detector.detect(df.iloc[max(0, i - history + 1): i + 1])
        res["lost_flow_lps"] = getattr(detector, "lost_flow_lps", None)
        rows.append(res)
    return pd.DataFrame(rows)


def _rate(n: int, d: int) -> float | None:
    return n / d if d else None


def score_holdout(results: pd.DataFrame, warmup: int) -> dict[str, Any]:
    """False-alert rate on normal data, skipping the warm-up rows."""
    results = results.iloc[warmup:]
    alerts = int(results["is_anomaly"].sum())
    return {"rows": len(results), "false_alerts": alerts, "false_alert_rate": _rate(alerts, len(results))}


def score_event(results: pd.DataFrame, case: dict[str, Any], warmup: int) -> dict[str, Any]:
    """Pre-event false alerts, event recall/delay/type/pipe, lost-flow estimate, post-event alerts."""
    truth = pd.read_csv(case["truth"])
    active = results[TIMESTAMP_COL].map(dict(zip(truth[TIMESTAMP_COL], truth["event_active"])))
    active = active.fillna(False).astype(bool)
    true_flow = results[TIMESTAMP_COL].map(dict(zip(truth[TIMESTAMP_COL], truth["leak_flow_m3s"]))) * 1000

    onset = results.loc[active, TIMESTAMP_COL].min()
    end = results.loc[active, TIMESTAMP_COL].max()
    pre = results[results[TIMESTAMP_COL] < onset].iloc[warmup:]
    during = results[active]
    post = results[results[TIMESTAMP_COL] > end]
    hits = during[during["is_anomaly"]]
    first = hits[TIMESTAMP_COL].min() if len(hits) else None

    out: dict[str, Any] = {
        "pre_rows": len(pre),
        "pre_false_alerts": int(pre["is_anomaly"].sum()),
        "event_rows": len(during),
        "recall": _rate(len(hits), len(during)),
        "first_alert_delay_min": (first - onset) / 60 if first is not None else None,
        "type_correct": _rate(int((hits["event_type"] == case["type"]).sum()), len(hits)),
        "red_share": _rate(int((hits["level"] == "red").sum()), len(hits)),
        "post_rows": len(post),
        "post_alerts": int(post["is_anomaly"].sum()),
        "types_seen": hits["event_type"].value_counts().to_dict(),
    }
    if len(hits) and hits["lost_flow_lps"].notna().any():
        out["lost_flow_est_lps"] = float(hits["lost_flow_lps"].median())
        out["lost_flow_true_lps"] = float(true_flow[hits.index].median())
    if case["pipe"]:
        pipe = case["pipe"]
        top2 = hits["pipe_scores"].map(lambda s: pipe in sorted(s, key=s.get, reverse=True)[:2])
        out["pipe_top1"] = _rate(int((hits["suspected_pipe"] == pipe).sum()), len(hits))
        out["pipe_top2"] = _rate(int(top2.sum()), len(hits))
        out["pipes_seen"] = hits["suspected_pipe"].value_counts().to_dict()
    return out


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{v:.0%}"


def evaluate_suite(suite: dict[str, Any], inp_path: Path, retrain: bool) -> dict[str, Any]:
    """Optionally train the suite's model, then score holdout and every case."""
    model = Path(suite["model"])
    if retrain or not model.exists():
        model.parent.mkdir(parents=True, exist_ok=True)
        train(suite["train"], model)
        print()
    warmup = LeakDetector(model, inp_path).window

    report: dict[str, Any] = {"holdout": score_holdout(run(suite["holdout"], model, inp_path), warmup)}
    for case in suite["cases"]:
        report[case["name"]] = score_event(run(case["csv"], model, inp_path), case, warmup)

    h = report["holdout"]
    print(f"=== {suite['name']} ===")
    print(f"Normal holdout: {h['false_alerts']}/{h['rows']} false alerts ({_pct(h['false_alert_rate'])})")
    print(f"{'case':<18}{'recall':>7}{'first':>8}{'type ok':>9}{'top-1':>7}{'top-2':>7}"
          f"{'pre FA':>9}{'lost L/s est/true':>20}  types / pipes")
    for case in suite["cases"]:
        e = report[case["name"]]
        delay = e["first_alert_delay_min"]
        flow = (f"{e['lost_flow_est_lps']:.2f}/{e['lost_flow_true_lps']:.2f}"
                if "lost_flow_est_lps" in e else "-")
        print(f"{case['name']:<18}{_pct(e['recall']):>7}{'-' if delay is None else f'{delay:.0f}m':>8}"
              f"{_pct(e['type_correct']):>9}{_pct(e.get('pipe_top1')):>7}{_pct(e.get('pipe_top2')):>7}"
              f"{e['pre_false_alerts']:>4}/{e['pre_rows']:<4}{flow:>20}  {e['types_seen']} {e.get('pipes_seen', '')}")
    print()
    return report


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Quick dev evaluation on simulated scenarios.")
    parser.add_argument("--suite", choices=["generated", "sensor_study", "all"], default="generated")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Model bundle for the generated suite.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR),
                        help="Where sensor-study models are saved / loaded.")
    parser.add_argument("--inp", default=DEFAULT_INP_PATH, help="EPANET network file.")
    parser.add_argument("--retrain", action="store_true", help="Retrain each suite's model first.")
    parser.add_argument("--json", help="Also write the report as JSON to this path.")
    args = parser.parse_args(argv)

    suites = []
    if args.suite in ("generated", "all"):
        suites.append(generated_suite(Path(args.model)))
    if args.suite in ("sensor_study", "all"):
        suites += [sensor_study_suite(v, Path(args.model_dir)) for v in ("clean", "noisy")]

    report = {s["name"]: evaluate_suite(s, Path(args.inp), args.retrain) for s in suites}
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
