"""Stream a scenario CSV through the AI detector and store everything in SQLite.

Usage:
    python -m backend.run_pipeline --data data/generated/rupture.csv
        [--model ai/model.joblib] [--inp data/Net1.inp] [--db data/aquawatch.db] [--speed 0]

Each row is fed to LeakDetector.detect() exactly as ai.detect.replay() does (the last
2*window-1 rows per step). Only the timestamp and sensor columns reach the detector;
scenario / leak_pipe go to the `truth` table for display only.
"""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ai.detect import LeakDetector
from ai.topology import DEFAULT_INP_PATH
from ai.train import DEFAULT_MODEL_PATH, LEAK_PIPE_COL, SCENARIO_COL, TIMESTAMP_COL, load_data

from . import db


def run_pipeline(
    data: str | Path,
    model: str | Path = DEFAULT_MODEL_PATH,
    inp: str | Path = DEFAULT_INP_PATH,
    db_path: str | Path = db.DEFAULT_DB_PATH,
    speed: float = 0.0,
    max_rows: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Replay `data` through the detector, saving readings and alerts. Returns a summary.

    speed: rows per second to simulate a live feed (0 = as fast as possible).
    max_rows: only process the first N rows (for tests / quick demos).
    avg detect time excludes the warm-up steps (no model call), as bench/ does.
    """
    df = load_data(data)
    if max_rows:
        df = df.head(max_rows)
    detector = LeakDetector(model, inp)
    history = 2 * detector.window - 1
    sensor_df = df[[TIMESTAMP_COL, *detector.sensor_cols]]  # labels never reach detect()
    has_truth = SCENARIO_COL in df.columns

    conn = db.init_db(db_path)
    writer = db.start_run(conn, data, model)
    latencies_ms = []
    alerts = 0
    t_start = time.monotonic()
    try:
        for i in range(len(df)):
            if speed > 0:  # pace to `speed` rows per second
                delay = t_start + i / speed - time.monotonic()
                if delay > 0:
                    time.sleep(delay)

            window_df = sensor_df.iloc[max(0, i - history + 1): i + 1]
            t0 = time.perf_counter_ns()
            result = detector.detect(window_df)
            elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
            if i >= detector.window - 1:
                latencies_ms.append(elapsed_ms)
            alerts += result["is_anomaly"]

            row = df.iloc[i]
            truth = None
            if has_truth:
                truth = {"scenario": row[SCENARIO_COL], "leak_pipe": row.get(LEAK_PIPE_COL)}
            db.save_step(writer, row, result, truth, lost_flow_lps=detector.lost_flow_lps)
    finally:  # an interrupted run is still closed properly with what it processed
        avg_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else None
        db.finish_run(writer, avg_ms)
        conn.close()

    summary = {
        "run_id": writer.run_id,
        "rows": writer.rows,
        "alerts": int(alerts),
        "avg_detect_ms": avg_ms,
        "platform": platform.machine(),
        "db": str(db_path),
    }
    if verbose:
        print("=== AquaWatch pipeline ===")
        print(f"Run id:        {summary['run_id']}  ({data})")
        print(f"Rows:          {summary['rows']}")
        print(f"Alert steps:   {summary['alerts']}")
        print(f"Avg detect():  {avg_ms:.2f} ms per step" if avg_ms is not None else "Avg detect():  -")
        print(f"Platform:      {summary['platform']}")
        print(f"Database:      {summary['db']}")
    return summary


def main(argv: Optional[list] = None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Run the AquaWatch detector on a CSV and store results in SQLite.")
    parser.add_argument("--data", required=True, help="Scenario sensor CSV.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Trained model bundle from ai.train.")
    parser.add_argument("--inp", default=DEFAULT_INP_PATH, help="EPANET network file.")
    parser.add_argument("--db", default=db.DEFAULT_DB_PATH, help="SQLite database file.")
    parser.add_argument("--speed", type=float, default=0.0,
                        help="Rows per second to simulate live data (0 = as fast as possible).")
    args = parser.parse_args(argv)
    run_pipeline(args.data, args.model, args.inp, args.db, args.speed)


if __name__ == "__main__":
    main()
