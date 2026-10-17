"""SQLite storage for AquaWatch sensor readings and AI alerts.

The pipeline (backend/run_pipeline.py) writes; the Streamlit dashboard reads.
Only the standard-library sqlite3 module plus pandas, so it runs anywhere Python
does, including IBM LinuxONE (s390x). WAL mode lets the dashboard read while the
pipeline is still writing.

Writing:
    conn = init_db("data/aquawatch.db")
    writer = start_run(conn, "data/generated/rupture.csv", "ai/model.joblib")
    save_step(writer, reading, alert, truth)   # once per timestep
    finish_run(writer, avg_detect_ms=12.3)

Reading (each call opens its own short-lived connection):
    list_runs(), get_readings(run_id), get_alerts(run_id),
    get_latest_alert(run_id, at_ts), get_kpis(run_id)
"""

from __future__ import annotations

import json
import platform
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import pandas as pd

DEFAULT_DB_PATH = "data/aquawatch.db"

# Sensor columns stored in `readings`. Must match ai.train (PRESSURE_COLS, the optional
# F_9 pump flow, FLOW_COLS); kept here so the dashboard doesn't need scikit-learn.
PRESSURE_COLS = [f"P_{j}" for j in (10, 11, 12, 13, 21, 22, 23, 31, 32)]
FLOW_COLS = [f"F_{p}" for p in (9, 10, 11, 12, 21, 22, 31, 110, 111, 112, 113, 121, 122)]
READING_COLS = PRESSURE_COLS + FLOW_COLS

BATCH_ROWS = 100       # flush buffered rows at least every 100 steps...
BATCH_SECONDS = 1.0    # ...or every second, so a slow live demo still shows up promptly

PathLike = Union[str, Path]
Timestamp = Union[int, float, str]

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS runs (
    run_id         INTEGER PRIMARY KEY,
    scenario_file  TEXT NOT NULL,
    model_path     TEXT NOT NULL,
    started_at     TEXT NOT NULL,
    finished_at    TEXT,
    rows_processed INTEGER NOT NULL DEFAULT 0,
    avg_detect_ms  REAL,
    platform       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS readings (
    run_id    INTEGER NOT NULL REFERENCES runs(run_id),
    timestamp NUMERIC NOT NULL,
    {", ".join(f"{c} REAL" for c in READING_COLS)},
    PRIMARY KEY (run_id, timestamp)
);
CREATE TABLE IF NOT EXISTS alerts (
    run_id         INTEGER NOT NULL REFERENCES runs(run_id),
    timestamp      NUMERIC NOT NULL,
    is_anomaly     INTEGER NOT NULL,
    severity       INTEGER NOT NULL,
    level          TEXT NOT NULL,
    suspected_pipe TEXT,
    event_type     TEXT NOT NULL,
    explanation    TEXT NOT NULL,
    lost_flow_lps  REAL,
    pipe_scores    TEXT NOT NULL,
    PRIMARY KEY (run_id, timestamp)
);
-- Ground truth for evaluation/display only; never fed to the model.
CREATE TABLE IF NOT EXISTS truth (
    run_id    INTEGER NOT NULL REFERENCES runs(run_id),
    timestamp NUMERIC NOT NULL,
    scenario  TEXT,
    leak_pipe TEXT,
    PRIMARY KEY (run_id, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_alerts_anomaly ON alerts(run_id, is_anomaly);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _db_ts(value: Any) -> Timestamp:
    """Timestamps are stored as numbers (elapsed seconds) or ISO strings."""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):  # numpy scalar
        value = value.item()
    return value


def _float_or_none(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return float(value)


def connect(path: PathLike = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a connection with WAL journaling (concurrent readers + one writer)."""
    conn = sqlite3.connect(str(path), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(path: PathLike = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Create the database file and tables if needed; return an open connection."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = connect(path)
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


# --------------------------------------------------------------------- writing

@dataclass
class RunWriter:
    """Buffers one run's rows and writes them in batches."""

    conn: sqlite3.Connection
    run_id: int
    rows: int = 0
    readings: List[tuple] = field(default_factory=list)
    alerts: List[tuple] = field(default_factory=list)
    truth: List[tuple] = field(default_factory=list)
    last_flush: float = field(default_factory=time.monotonic)

    def flush(self) -> None:
        """Insert all buffered rows and commit."""
        cols = ", ".join(READING_COLS)
        marks = ", ".join("?" * (len(READING_COLS) + 2))
        with self.conn:  # commits on success
            self.conn.executemany(
                f"INSERT INTO readings (run_id, timestamp, {cols}) VALUES ({marks})", self.readings)
            self.conn.executemany(
                "INSERT INTO alerts (run_id, timestamp, is_anomaly, severity, level, suspected_pipe, "
                "event_type, explanation, lost_flow_lps, pipe_scores) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                self.alerts)
            self.conn.executemany(
                "INSERT INTO truth (run_id, timestamp, scenario, leak_pipe) VALUES (?, ?, ?, ?)", self.truth)
            self.conn.execute("UPDATE runs SET rows_processed = ? WHERE run_id = ?", (self.rows, self.run_id))
        self.readings.clear()
        self.alerts.clear()
        self.truth.clear()
        self.last_flush = time.monotonic()


def start_run(conn: sqlite3.Connection, scenario_file: PathLike, model_path: PathLike) -> RunWriter:
    """Register a new run and return a writer for its steps."""
    with conn:
        cur = conn.execute(
            "INSERT INTO runs (scenario_file, model_path, started_at, platform) VALUES (?, ?, ?, ?)",
            (str(scenario_file), str(model_path), _now(), platform.machine()),
        )
    return RunWriter(conn=conn, run_id=int(cur.lastrowid))


def save_step(
    writer: RunWriter,
    reading: Mapping[str, Any],
    alert: Mapping[str, Any],
    truth: Optional[Mapping[str, Any]] = None,
    lost_flow_lps: Optional[float] = None,
) -> None:
    """Buffer one timestep: the raw sensor reading, the detect() result and optional truth.

    reading needs the timestamp and sensor values (missing sensors are stored as NULL).
    alert is the dict returned by LeakDetector.detect(). truth may hold
    scenario / leak_pipe for display; it is never used by the model.
    """
    ts = _db_ts(alert["timestamp"])
    writer.readings.append((writer.run_id, ts, *(_float_or_none(reading.get(c)) for c in READING_COLS)))
    writer.alerts.append((
        writer.run_id, ts, int(bool(alert["is_anomaly"])), int(alert["severity"]), alert["level"],
        alert["suspected_pipe"], alert["event_type"], alert["explanation"],
        _float_or_none(lost_flow_lps), json.dumps(alert["pipe_scores"]),
    ))
    if truth is not None:
        leak = truth.get("leak_pipe")
        writer.truth.append((
            writer.run_id, ts, truth.get("scenario"),
            None if leak is None or pd.isna(leak) or leak == "" else str(leak),
        ))
    writer.rows += 1
    if len(writer.alerts) >= BATCH_ROWS or time.monotonic() - writer.last_flush >= BATCH_SECONDS:
        writer.flush()


def finish_run(writer: RunWriter, avg_detect_ms: Optional[float] = None) -> None:
    """Write any buffered rows and record the run's end time and stats."""
    writer.flush()
    with writer.conn:
        writer.conn.execute(
            "UPDATE runs SET finished_at = ?, rows_processed = ?, avg_detect_ms = ? WHERE run_id = ?",
            (_now(), writer.rows, avg_detect_ms, writer.run_id),
        )


# --------------------------------------------------------------------- reading

def _query(db_path: PathLike, sql: str, params: tuple = ()) -> pd.DataFrame:
    conn = connect(db_path)
    try:
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()


def list_runs(db_path: PathLike = DEFAULT_DB_PATH) -> pd.DataFrame:
    """All runs, newest first."""
    return _query(db_path, "SELECT * FROM runs ORDER BY run_id DESC")


def get_readings(run_id: int, until_ts: Optional[Timestamp] = None,
                 db_path: PathLike = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Sensor readings of a run in time order, optionally only up to until_ts (inclusive)."""
    run_id = int(run_id)  # numpy int64 (e.g. from list_runs) silently matches nothing in sqlite3
    sql = "SELECT * FROM readings WHERE run_id = ?"
    params: tuple = (run_id,)
    if until_ts is not None:
        sql += " AND timestamp <= ?"
        params += (_db_ts(until_ts),)
    return _query(db_path, sql + " ORDER BY timestamp", params)


def get_alerts(run_id: int, until_ts: Optional[Timestamp] = None, only_anomalies: bool = False,
               db_path: PathLike = DEFAULT_DB_PATH) -> pd.DataFrame:
    """detect() results of a run in time order; pipe_scores decoded back to dicts."""
    run_id = int(run_id)
    sql = "SELECT * FROM alerts WHERE run_id = ?"
    params: tuple = (run_id,)
    if until_ts is not None:
        sql += " AND timestamp <= ?"
        params += (_db_ts(until_ts),)
    if only_anomalies:
        sql += " AND is_anomaly = 1"
    df = _query(db_path, sql + " ORDER BY timestamp", params)
    df["is_anomaly"] = df["is_anomaly"].astype(bool)
    df["pipe_scores"] = df["pipe_scores"].map(json.loads)
    return df


def get_latest_alert(run_id: int, at_ts: Optional[Timestamp] = None,
                     db_path: PathLike = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """The detect() result at or just before at_ts (latest if None), as detect() returns it.

    Returns None if the run has no results yet.
    """
    run_id = int(run_id)
    sql = "SELECT * FROM alerts WHERE run_id = ?"
    params: tuple = (run_id,)
    if at_ts is not None:
        sql += " AND timestamp <= ?"
        params += (_db_ts(at_ts),)
    df = _query(db_path, sql + " ORDER BY timestamp DESC LIMIT 1", params)
    if df.empty:
        return None
    row = df.iloc[0]
    ts = row["timestamp"]
    return {
        "timestamp": ts.item() if hasattr(ts, "item") else ts,
        "is_anomaly": bool(row["is_anomaly"]),
        "severity": int(row["severity"]),
        "level": row["level"],
        "suspected_pipe": None if pd.isna(row["suspected_pipe"]) else row["suspected_pipe"],
        "pipe_scores": json.loads(row["pipe_scores"]),
        "event_type": row["event_type"],
        "explanation": row["explanation"],
    }


def get_kpis(run_id: int, db_path: PathLike = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Headline numbers for a run: steps, alerts, first alert time, speed, platform."""
    run_id = int(run_id)
    df = _query(
        db_path,
        "SELECT r.scenario_file, r.platform, r.avg_detect_ms, r.started_at, r.finished_at, "
        "COUNT(a.timestamp) AS total_steps, COALESCE(SUM(a.is_anomaly), 0) AS alert_count, "
        "MIN(CASE WHEN a.is_anomaly = 1 THEN a.timestamp END) AS first_alert_time "
        "FROM runs r LEFT JOIN alerts a ON a.run_id = r.run_id WHERE r.run_id = ? GROUP BY r.run_id",
        (run_id,),
    )
    if df.empty:
        raise KeyError(f"No run with run_id={run_id}")
    row = df.iloc[0]
    first = row["first_alert_time"]
    return {
        "run_id": run_id,
        "scenario_file": row["scenario_file"],
        "total_steps": int(row["total_steps"]),
        "alert_count": int(row["alert_count"]),
        "first_alert_time": None if pd.isna(first) else (first.item() if hasattr(first, "item") else first),
        "avg_detect_ms": _float_or_none(row["avg_detect_ms"]),
        "platform": row["platform"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
    }
