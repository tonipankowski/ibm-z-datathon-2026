"""Forest-only steps are reported as data/sensor warnings, never as leak alerts."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pytest

from ai import detect
from ai.train import train

ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = ROOT / "data/generated/normal_train.csv"
HOLDOUT_CSV = ROOT / "data/generated/normal_holdout.csv"
INP = ROOT / "data/Net1.inp"
KEYS = ["timestamp", "is_anomaly", "severity", "level", "suspected_pipe",
        "pipe_scores", "event_type", "explanation"]


def _alert(**overrides):
    alert = {"timestamp": 0, "is_anomaly": False, "severity": 0, "level": "green",
             "suspected_pipe": None, "pipe_scores": {}, "event_type": "normal",
             "explanation": "All sensors within normal range."}
    alert.update(overrides)
    return alert


def test_frontend_prefix_matches_detector():
    db_view = pytest.importorskip("frontend.db_view")
    assert db_view.DATA_WARNING_PREFIX == detect.DATA_WARNING_PREFIX


def test_alert_status_cases():
    db_view = pytest.importorskip("frontend.db_view")
    warning = _alert(severity=52, level="yellow",
                     explanation=f"{detect.DATA_WARNING_PREFIX} the anomaly model flags ...")
    assert db_view.alert_status(warning) == ("DATA WARNING · check sensors", "data warning", "info")
    watch = _alert(severity=45, level="yellow", explanation="Unusual readings for 1 step(s) ...")
    assert db_view.alert_status(watch)[0].startswith("WATCH")
    leak = _alert(is_anomaly=True, severity=90, level="red", event_type="leak", suspected_pipe="12")
    assert db_view.alert_status(leak)[2] == "error"
    assert db_view.alert_status(_alert())[0] == "GREEN · normal"


@pytest.mark.skipif(not (TRAIN_CSV.exists() and HOLDOUT_CSV.exists()), reason="data/generated/ not available")
def test_data_warnings_on_holdout_keep_schema(tmp_path):
    model = tmp_path / "model.joblib"
    with contextlib.redirect_stdout(io.StringIO()):
        train(TRAIN_CSV, model)
    results = detect.replay(HOLDOUT_CSV, model, INP, verbose=False)
    warnings = [r for r in results if r["explanation"].startswith(detect.DATA_WARNING_PREFIX)]
    assert warnings, "expected the forest to flag at least one holdout step"
    for r in warnings:
        assert list(r) == KEYS
        assert r["is_anomaly"] is False
        assert r["event_type"] == "normal"
        assert r["suspected_pipe"] is None
        assert r["level"] == "yellow"
        assert "Check sensors:" in r["explanation"]
