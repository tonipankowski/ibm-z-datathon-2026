"""Performance benchmark for the AquaWatch AI pipeline.

Usage:
    python -m bench.benchmark [--repeats 3] [--scenarios leak rupture] [--max-rows 500]

Measures speed and resource use, not detection accuracy (see EVALUATION.md):
1. Training time and memory on normal_train.csv.
2. Per-step detect() latency (p50/p95/p99) and throughput while replaying each
   scenario CSV row by row, exactly as a live sensor stream would arrive.

Run the same command on a laptop and on LinuxONE (s390x) and compare the JSON
files written to bench/results/.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import socket
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import networkx
import numpy as np
import pandas as pd
import sklearn

try:  # optional: may not build on s390x without a compiler
    import psutil
except ImportError:
    psutil = None

from ai.detect import LeakDetector
from ai.topology import DEFAULT_INP_PATH
from ai.train import load_data, train

DATA_DIR = Path("data/generated")
TRAIN_CSV = DATA_DIR / "normal_train.csv"
DEFAULT_SCENARIOS = ["normal_holdout", "leak", "rupture", "demand_spike"]
DEFAULT_REPEATS = 3
DEFAULT_OUT_DIR = Path("bench/results")


def _mb(n_bytes: float) -> float:
    return round(n_bytes / 1024 / 1024, 1)


def _rss_mb() -> float | None:
    """Current resident memory of this process, in MB (None if unavailable)."""
    if psutil is not None:
        return _mb(psutil.Process().memory_info().rss)
    try:  # Linux without psutil
        with open("/proc/self/statm") as f:
            return _mb(int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, AttributeError):
        return None


def _peak_rss_mb() -> float | None:
    """Peak resident memory of this process so far, in MB (None if unavailable)."""
    if psutil is not None:
        info = psutil.Process().memory_info()
        if hasattr(info, "peak_wset"):  # Windows
            return _mb(info.peak_wset)
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KB, macOS reports bytes.
    return _mb(peak if sys.platform == "darwin" else peak * 1024)


def system_info() -> dict[str, Any]:
    """Describe the machine and library versions, so results can be compared fairly."""
    freq = psutil.cpu_freq() if psutil else None
    if psutil:
        ram_total = psutil.virtual_memory().total
    else:
        try:  # Linux without psutil
            ram_total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (ValueError, AttributeError, OSError):
            ram_total = None
    return {
        "hostname": socket.gethostname(),
        "machine": platform.machine(),  # e.g. AMD64, arm64, s390x
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "cpu_logical": psutil.cpu_count(logical=True) if psutil else os.cpu_count(),
        "cpu_physical": psutil.cpu_count(logical=False) if psutil else None,
        "cpu_max_mhz": round(freq.max) if freq and freq.max else None,
        "ram_total_mb": _mb(ram_total) if ram_total else None,
        "psutil": psutil.__version__ if psutil else None,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "networkx": networkx.__version__,
        "joblib": joblib.__version__,
    }


def bench_train(model_path: Path, repeats: int) -> dict[str, Any]:
    """Time train() on the normal training CSV, `repeats` times."""
    times = []
    for _ in range(repeats):
        with contextlib.redirect_stdout(io.StringIO()):  # train() prints a summary
            start = time.perf_counter()
            train(TRAIN_CSV, model_path)
            times.append(time.perf_counter() - start)
    return {
        "data": str(TRAIN_CSV),
        "rows": len(pd.read_csv(TRAIN_CSV)),
        "repeats": repeats,
        "seconds_median": round(statistics.median(times), 3),
        "seconds_min": round(min(times), 3),
        "seconds_all": [round(t, 3) for t in times],
        "model_file_kb": round(model_path.stat().st_size / 1024, 1),
    }


def _percentiles_ms(latencies_ns: list[int]) -> dict[str, float]:
    ms = np.array(latencies_ns) / 1e6
    p50, p95, p99 = np.percentile(ms, [50, 95, 99])
    return {
        "mean": round(float(ms.mean()), 3),
        "p50": round(float(p50), 3),
        "p95": round(float(p95), 3),
        "p99": round(float(p99), 3),
        "max": round(float(ms.max()), 3),
    }


def bench_detect(scenario: str, model_path: Path, max_rows: int | None) -> dict[str, Any]:
    """Replay one scenario CSV through LeakDetector.detect(), timing every step."""
    csv_path = DATA_DIR / f"{scenario}.csv"
    df = load_data(csv_path)
    if max_rows:
        df = df.head(max_rows)

    start = time.perf_counter()
    detector = LeakDetector(model_path, DEFAULT_INP_PATH)
    load_seconds = time.perf_counter() - start

    history = 2 * detector.window - 1  # same history replay() passes in
    latencies_ns: list[int] = []
    alerts = 0
    total_start = time.perf_counter()
    for i in range(len(df)):
        window_df = df.iloc[max(0, i - history + 1): i + 1]
        t0 = time.perf_counter_ns()
        result = detector.detect(window_df)
        latencies_ns.append(time.perf_counter_ns() - t0)
        alerts += result["is_anomaly"]
    total_seconds = time.perf_counter() - total_start

    # Skip warm-up steps (early return, no model call) so they don't flatter the numbers.
    scored = latencies_ns[detector.window - 1:] or latencies_ns
    return {
        "data": str(csv_path),
        "rows": len(df),
        "detector_load_seconds": round(load_seconds, 3),
        "total_seconds": round(total_seconds, 3),
        "rows_per_second": round(len(df) / total_seconds, 1),
        "latency_ms": _percentiles_ms(scored),
        "alerting_steps": int(alerts),
    }


def run(scenarios: list[str], repeats: int, max_rows: int | None, out_dir: Path) -> dict[str, Any]:
    """Run all benchmarks, save the JSON report and return it."""
    report: dict[str, Any] = {
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "system": system_info(),
        "rss_mb_start": _rss_mb(),
    }

    with tempfile.TemporaryDirectory() as tmp:  # don't overwrite the team's ai/model.joblib
        model_path = Path(tmp) / "model_bench.joblib"
        print(f"Training on {TRAIN_CSV} ({repeats}x)...")
        report["train"] = bench_train(model_path, repeats)
        report["rss_mb_after_train"] = _rss_mb()

        report["detect"] = {}
        for scenario in scenarios:
            print(f"Replaying {scenario}...")
            report["detect"][scenario] = bench_detect(scenario, model_path, max_rows)

    report["rss_mb_end"] = _rss_mb()
    report["peak_rss_mb"] = _peak_rss_mb()

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"{report['system']['machine'].lower()}_{report['system']['hostname']}_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["saved_to"] = str(out_path)
    return report


def print_summary(report: dict[str, Any]) -> None:
    sysinfo, tr = report["system"], report["train"]
    print()
    print("=== AquaWatch benchmark ===")
    ram = f"{sysinfo['ram_total_mb']:.0f} MB" if sysinfo["ram_total_mb"] else "unknown"
    print(f"Machine:   {sysinfo['machine']} | {sysinfo['cpu_logical']} logical CPUs | "
          f"{ram} RAM | Python {sysinfo['python']}")
    print(f"Training:  {tr['seconds_median']:.3f} s median over {tr['repeats']} runs "
          f"({tr['rows']} rows, model {tr['model_file_kb']} KB)")
    print()
    print(f"{'Scenario':<16}{'rows':>6}{'rows/s':>10}{'p50 ms':>9}{'p95 ms':>9}{'p99 ms':>9}{'max ms':>9}")
    for name, d in report["detect"].items():
        lat = d["latency_ms"]
        print(f"{name:<16}{d['rows']:>6}{d['rows_per_second']:>10.1f}"
              f"{lat['p50']:>9.2f}{lat['p95']:>9.2f}{lat['p99']:>9.2f}{lat['max']:>9.2f}")
    print()
    print(f"Memory:    {report['rss_mb_start']} MB at start, {report['rss_mb_end']} MB at end, "
          f"peak {report['peak_rss_mb']} MB")
    print(f"Saved to:  {report['saved_to']}")


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Benchmark AquaWatch training and detection speed.")
    parser.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS,
                        help="Scenario CSV names in data/generated/ (without .csv).")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help="Training runs to time.")
    parser.add_argument("--max-rows", type=int, default=None, help="Replay only the first N rows (quick test).")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Where to save the JSON report.")
    args = parser.parse_args(argv)

    # Run from the repo root so relative data paths resolve.
    os.chdir(Path(__file__).resolve().parent.parent)
    print_summary(run(args.scenarios, args.repeats, args.max_rows, args.out_dir))


if __name__ == "__main__":
    main()
