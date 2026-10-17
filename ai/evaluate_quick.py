"""Quick developer check of the detector on Role 1's generated scenarios.

Not the official evaluation (Role 3 owns that). Replays each CSV through
LeakDetector and scores the results against the *.truth.csv event windows.
Truth files and the scenario/leak_pipe columns are used here only for scoring.

Usage:
    python -m ai.evaluate_quick [--model ai/model.joblib] [--retrain] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .detect import LeakDetector, replay
from .topology import DEFAULT_INP_PATH
from .train import DEFAULT_MODEL_PATH, TIMESTAMP_COL, train

DATA_DIR = Path("data/generated")
TRAIN_CSV = DATA_DIR / "normal_train.csv"
HOLDOUT_CSV = DATA_DIR / "normal_holdout.csv"
TRUE_PIPE = "12"
EVENTS = {  # file stem -> expected event_type
    "leak": "leak",
    "rupture": "rupture",
    "demand_spike": "demand_spike",
}


def _rate(n: int, d: int) -> float | None:
    return n / d if d else None


def evaluate_holdout(model_path: Path, inp_path: Path, warmup: int) -> dict[str, Any]:
    """False-alert rate on the normal holdout, skipping the warm-up rows."""
    results = replay(HOLDOUT_CSV, model_path, inp_path, verbose=False)[warmup:]
    alerts = sum(r["is_anomaly"] for r in results)
    return {"rows": len(results), "false_alerts": alerts, "false_alert_rate": _rate(alerts, len(results))}


def evaluate_event(stem: str, expected: str, model_path: Path, inp_path: Path, warmup: int) -> dict[str, Any]:
    """Pre-event false alerts, event recall/delay/localization/type, post-event alerts."""
    results = pd.DataFrame(replay(DATA_DIR / f"{stem}.csv", model_path, inp_path, verbose=False))
    truth = pd.read_csv(DATA_DIR / f"{stem}.truth.csv")
    active = results[TIMESTAMP_COL].map(dict(zip(truth[TIMESTAMP_COL], truth["event_active"]))).fillna(False).astype(bool)

    onset = results.loc[active, TIMESTAMP_COL].min()
    end = results.loc[active, TIMESTAMP_COL].max()
    pre = results[(results[TIMESTAMP_COL] < onset)].iloc[warmup:]
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
        "type_correct": _rate(int((hits["event_type"] == expected).sum()), len(hits)),
        "red_share": _rate(int((hits["level"] == "red").sum()), len(hits)),
        "post_rows": len(post),
        "post_alerts": int(post["is_anomaly"].sum()),
        "types_seen": hits["event_type"].value_counts().to_dict(),
    }
    if expected != "demand_spike":
        top2 = hits["pipe_scores"].map(lambda s: TRUE_PIPE in sorted(s, key=s.get, reverse=True)[:2])
        out["pipe12_top1"] = _rate(int((hits["suspected_pipe"] == TRUE_PIPE).sum()), len(hits))
        out["pipe12_top2"] = _rate(int(top2.sum()), len(hits))
        out["pipes_seen"] = hits["suspected_pipe"].value_counts().to_dict()
    return out


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.1%}" if v <= 1 else f"{v:.1f}"
    return str(v)


def evaluate(model_path: Path, inp_path: Path) -> dict[str, Any]:
    """Run all checks and print a compact report."""
    warmup = LeakDetector(model_path, inp_path).window
    report: dict[str, Any] = {"holdout": evaluate_holdout(model_path, inp_path, warmup)}
    for stem, expected in EVENTS.items():
        report[stem] = evaluate_event(stem, expected, model_path, inp_path, warmup)

    h = report["holdout"]
    print(f"Normal holdout: {h['false_alerts']}/{h['rows']} false alerts ({_fmt(h['false_alert_rate'])})")
    for stem in EVENTS:
        e = report[stem]
        delay = e["first_alert_delay_min"]
        print(
            f"{stem:<13} pre-event FA {e['pre_false_alerts']}/{e['pre_rows']} | "
            f"recall {_fmt(e['recall'])} | first alert "
            f"{'missed' if delay is None else f'+{delay:.0f} min'} | "
            f"type ok {_fmt(e['type_correct'])} {e['types_seen']} | red {_fmt(e['red_share'])} | "
            f"post alerts {e['post_alerts']}/{e['post_rows']}"
        )
        if "pipe12_top1" in e:
            print(f"{'':<13} pipe 12 top-1 {_fmt(e['pipe12_top1'])} | top-2 {_fmt(e['pipe12_top2'])} | {e['pipes_seen']}")
    return report


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Quick dev evaluation on data/generated.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Model bundle to evaluate.")
    parser.add_argument("--inp", default=DEFAULT_INP_PATH, help="EPANET network file.")
    parser.add_argument("--retrain", action="store_true", help=f"Train on {TRAIN_CSV} first.")
    parser.add_argument("--json", help="Also write the report as JSON to this path.")
    args = parser.parse_args(argv)

    if args.retrain:
        train(TRAIN_CSV, args.model)
        print()
    report = evaluate(Path(args.model), Path(args.inp))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
