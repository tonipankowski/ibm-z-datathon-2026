from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from ai.detect import LeakDetector
from ai.topology import load_network
from ai.train import train
from frontend.charts import network_figure, sensor_figure
from frontend.replay import DATA, ROOT, SCENARIOS, Replay


@pytest.fixture(scope="session")
def model(tmp_path_factory):
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    train(DATA / "normal/train.clean.csv", path)
    return path


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_real_detector_seek_and_truth_isolation(model, scenario, tmp_path):
    source = SCENARIOS[scenario]
    # Include pre-event, event and recovery rows without a seven-day test cost.
    frame = pd.read_csv(source).iloc[995:1095].reset_index(drop=True)
    original = tmp_path / "scenario.csv"
    frame.to_csv(original, index=False)
    replay = Replay(original, model, ROOT / "data/Net1.inp")
    reference = LeakDetector(model, ROOT / "data/Net1.inp")
    expected = [reference.detect(replay.readings.iloc[max(0, i-18):i+1]) for i in range(len(frame))]
    assert replay.at(90) == expected[90]
    assert replay.at(20) == expected[20]
    assert replay.at(99) == expected[99]
    assert replay.results == expected
    frame["scenario"] = "deliberately_wrong"
    frame["leak_pipe"] = "not-a-pipe"
    changed = tmp_path / "changed.csv"
    frame.to_csv(changed, index=False)
    other = Replay(changed, model, ROOT / "data/Net1.inp")
    other.at(99)
    assert other.results == expected
    assert "scenario" not in replay.readings
    assert "leak_pipe" not in replay.readings


def test_map_and_units(model):
    replay = Replay(SCENARIOS["Normal"], model, ROOT / "data/Net1.inp")
    graph = load_network(ROOT / "data/Net1.inp")
    fig = network_figure(graph, replay.frame.iloc[0], "P_10", "F_9", "12")
    names = {t.name for t in fig.data}
    assert {"Pump 9", "Tank", "Reservoir", "Junction"} <= names
    assert len([t for t in fig.data if t.mode == "lines"]) == graph.number_of_edges()
    chart = sensor_figure(replay.frame, 10, "F_9", "Flow (m³/s, signed)")
    assert len(chart.data[0].x) == 11
    assert "m³/s" in chart.layout.yaxis.title.text


def test_app_controls_and_scenario_reset(model, monkeypatch):
    monkeypatch.setenv("AQUAWATCH_MODEL", str(model))
    app = AppTest.from_file(str(ROOT / "dashboard.py"), default_timeout=60).run()
    assert not app.exception
    app.button[0].click().run()
    assert app.session_state["playing"]
    assert app.session_state["position"] > 0
    app.button[0].click().run()
    assert not app.session_state["playing"]
    app.slider[0].set_value(25).run()
    assert app.session_state["position"] == 25
    app.slider[0].set_value(5).run()
    assert app.session_state["position"] == 5
    for name in SCENARIOS:
        app.sidebar.selectbox[0].select(name).run()
        assert not app.exception
        assert app.session_state["position"] == (5 if name == "Normal" else 0)
    app.sidebar.selectbox[1].select("P_22").run()
    app.sidebar.selectbox[2].select("F_9").run()
    assert not app.exception
    app.slider[0].set_value(2016).run(timeout=120)
    assert not app.session_state["playing"]
    app.button[0].click().run()
    assert app.session_state["position"] == 12
    app.button[0].click().run()
    assert not app.exception


def test_missing_model_is_explicit(monkeypatch, tmp_path):
    monkeypatch.setenv("AQUAWATCH_MODEL", str(tmp_path / "missing.joblib"))
    app = AppTest.from_file(str(ROOT / "dashboard.py")).run()
    assert not app.exception
    assert "Detector model missing" in app.error[0].value
    assert not app.metric
