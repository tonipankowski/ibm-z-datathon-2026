"""Tests for backend.db and backend.run_pipeline on a temporary database."""

from __future__ import annotations

import contextlib
import io
import platform
from pathlib import Path

import pytest

from ai.detect import LeakDetector
from ai.topology import DEFAULT_INP_PATH
from ai.train import FLOW_COLS, OPTIONAL_SENSOR_COLS, PRESSURE_COLS, load_data, train
from backend import db
from backend.run_pipeline import run_pipeline

ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = ROOT / "data/generated/normal_train.csv"
RUPTURE_CSV = ROOT / "data/generated/rupture.csv"
INP = ROOT / DEFAULT_INP_PATH
N_ROWS = 200

pytestmark = pytest.mark.skipif(
    not (TRAIN_CSV.exists() and RUPTURE_CSV.exists()), reason="data/generated/ not available"
)


@pytest.fixture(scope="module")
def model_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    with contextlib.redirect_stdout(io.StringIO()):
        train(TRAIN_CSV, path)
    return path


@pytest.fixture(scope="module")
def run(model_path, tmp_path_factory):
    db_path = tmp_path_factory.mktemp("db") / "test.db"
    summary = run_pipeline(RUPTURE_CSV, model_path, INP, db_path, max_rows=N_ROWS, verbose=False)
    return summary, db_path


def test_reading_columns_match_model_sensors():
    assert set(db.READING_COLS) == set(PRESSURE_COLS + FLOW_COLS + OPTIONAL_SENSOR_COLS)


def test_runs_and_kpis(run):
    summary, db_path = run
    runs = db.list_runs(db_path)
    assert len(runs) == 1
    assert runs.loc[0, "run_id"] == summary["run_id"]
    assert runs.loc[0, "rows_processed"] == N_ROWS
    assert runs.loc[0, "platform"] == platform.machine()
    assert runs.loc[0, "finished_at"] is not None

    kpis = db.get_kpis(summary["run_id"], db_path)
    assert kpis["total_steps"] == N_ROWS
    assert kpis["alert_count"] == summary["alerts"]
    assert kpis["avg_detect_ms"] > 0
    assert kpis["platform"] == platform.machine()


def test_read_functions_accept_run_id_from_list_runs(run):
    summary, db_path = run
    run_id = db.list_runs(db_path)["run_id"].iloc[0]  # numpy int64, as the frontend gets it
    assert len(db.get_readings(run_id, db_path=db_path)) == N_ROWS
    assert len(db.get_alerts(run_id, db_path=db_path)) == N_ROWS
    assert db.get_latest_alert(run_id, db_path=db_path) is not None
    kpis = db.get_kpis(run_id, db_path=db_path)
    assert kpis["total_steps"] == N_ROWS
    assert type(kpis["run_id"]) is int


def test_readings_shape_and_until(run):
    summary, db_path = run
    readings = db.get_readings(summary["run_id"], db_path=db_path)
    assert readings.shape == (N_ROWS, 2 + len(db.READING_COLS))
    assert readings["timestamp"].is_monotonic_increasing

    source = load_data(RUPTURE_CSV).head(N_ROWS)
    assert readings["P_13"].tolist() == pytest.approx(source["P_13"].tolist())

    cutoff = source["timestamp"].iloc[49]
    assert len(db.get_readings(summary["run_id"], until_ts=cutoff, db_path=db_path)) == 50


def test_alerts_decoded(run):
    summary, db_path = run
    alerts = db.get_alerts(summary["run_id"], db_path=db_path)
    assert len(alerts) == N_ROWS
    scores = alerts["pipe_scores"].iloc[-1]
    assert isinstance(scores, dict) and len(scores) == 12
    assert sum(scores.values()) == pytest.approx(1.0)
    anomalies = db.get_alerts(summary["run_id"], only_anomalies=True, db_path=db_path)
    assert len(anomalies) == summary["alerts"]
    assert anomalies["is_anomaly"].all()


def test_latest_alert_matches_detect(run, model_path):
    summary, db_path = run
    # Replay the same rows directly, as the pipeline does.
    df = load_data(RUPTURE_CSV).head(N_ROWS)
    detector = LeakDetector(model_path, INP)
    history = 2 * detector.window - 1
    for i in range(len(df)):
        expected = detector.detect(df.iloc[max(0, i - history + 1): i + 1])

    latest = db.get_latest_alert(summary["run_id"], db_path=db_path)
    assert list(latest) == list(expected)  # same keys, same order
    assert latest == expected

    at = db.get_latest_alert(summary["run_id"], at_ts=df["timestamp"].iloc[100] + 1, db_path=db_path)
    assert at["timestamp"] == df["timestamp"].iloc[100]
    assert db.get_latest_alert(summary["run_id"], at_ts=-1, db_path=db_path) is None
