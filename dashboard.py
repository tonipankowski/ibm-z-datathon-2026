"""Run from the repository root: python -m streamlit run dashboard.py."""
import os
from pathlib import Path

import streamlit as st

from ai.topology import load_network
from frontend.charts import network_figure, sensor_figure
from frontend.replay import ROOT, SCENARIOS, Replay

st.set_page_config(page_title="AquaWatch | Simulated network", page_icon="💧", layout="wide")
st.title("💧 AquaWatch")
st.caption("SIMULATED DATA · Net1 water network · Local scenario replay")
st.warning("Experimental detector: false alarms occur. Severity and localization confidence are heuristic scores, not proven accuracy or calibrated probabilities.")

model_path = Path(os.environ.get("AQUAWATCH_MODEL", str(ROOT / "ai/model.joblib")))
inp_path = ROOT / "data/Net1.inp"
scenario = st.sidebar.selectbox("Simulated scenario", list(SCENARIOS))
st.sidebar.caption("Clean EPANET simulation · 5-minute samples · 7 days. Leak and rupture use pipe 12; rupture is a large orifice, not complete severance.")
st.sidebar.caption("Pressure: metres of head. Flow: signed m³/s in original link orientation.")
if not model_path.exists():
    st.error("Detector model missing. Train the normal baseline before replay; no substitute alerts are generated.")
    st.code("python -m ai.train --data data/sensor_study/normal/train.clean.csv --out ai/model.joblib")
    st.stop()

csv_path = SCENARIOS[scenario]
try:
    identity = (scenario, csv_path.stat().st_mtime_ns, str(model_path.resolve()), model_path.stat().st_mtime_ns, inp_path.stat().st_mtime_ns)
    if st.session_state.get("identity") != identity:
        st.session_state.replay = Replay(csv_path, model_path, inp_path)
        st.session_state.identity = identity
        st.session_state.position = 0
        st.session_state.playing = False
    replay = st.session_state.replay
    graph = load_network(inp_path)
except Exception as exc:
    st.error(f"Cannot initialize real detector replay: {exc}")
    st.stop()

pressure = st.sidebar.selectbox("Pressure sensor (junction)", [c for c in replay.detector.sensor_cols if c.startswith("P_")])
flow = st.sidebar.selectbox("Flow sensor (pipe / pump)", [c for c in replay.detector.sensor_cols if c.startswith("F_")])
step = st.sidebar.select_slider("Readings per playback tick (1 second)", options=[1, 5, 12, 60], value=12)
st.sidebar.caption(f"Source: {csv_path.relative_to(ROOT)}")


def toggle():
    if st.session_state.position == len(replay.frame)-1:
        st.session_state.position = 0
    st.session_state.playing = not st.session_state.playing


def pause():
    st.session_state.playing = False
    st.session_state.seeked = True


st.button("Pause" if st.session_state.playing else "Play", on_click=toggle)


@st.fragment(run_every=1 if st.session_state.playing else None)
def render_replay():
    if st.session_state.pop("seeked", False):
        st.rerun()
    if st.session_state.playing:
        st.session_state.position = min(st.session_state.position + step, len(replay.frame)-1)
        if st.session_state.position == len(replay.frame)-1:
            st.session_state.playing = False
            st.rerun()
    index = st.slider("Replay time (reading index; elapsed time below)", 0, len(replay.frame)-1,
                      key="position", on_change=pause)
    row = replay.frame.iloc[index]
    seconds = int(row.timestamp)
    st.caption(f"Simulated elapsed time: day {seconds//86400}, {(seconds%86400)//3600:02d}:{(seconds%3600)//60:02d} · {seconds:,} s · Reading {index+1:,}/{len(replay.frame):,}")
    try:
        with st.spinner("Processing real detector readings in chronological order…"):
            result = replay.at(index)
    except Exception as exc:
        st.error(f"Detector unavailable at this reading: {exc}. No prediction is substituted.")
        return
    network, alerts = st.columns([1.6, 1])
    with network:
        st.subheader("Simulated network")
        st.plotly_chart(network_figure(graph, row, pressure, flow, result["suspected_pipe"]), width="stretch")
        st.caption("Blue: selected flow sensor · Larger junction: selected pressure sensor · Red: detector-suspected pipe · Dashed purple: pump. Layout uses Net1 model coordinates, not a geographic map.")
    with alerts:
        st.subheader("Detector prediction")
        a, b = st.columns(2)
        a.metric("Severity (heuristic / 100)", result["severity"])
        b.metric("Detector level", result["level"].upper())
        st.write("**Event type:**", result["event_type"])
        st.write("**Suspected pipe:**", result["suspected_pipe"] or "None")
        st.write("**Confirmed anomaly by detector:**", "Yes" if result["is_anomaly"] else "No")
        st.info(result["explanation"])
        st.caption("Explanation is the detector's original output. Any ‘confidence’ wording describes relative localization weights, not demonstrated correctness. Initial readings include model warm-up.")
        st.divider()
        st.subheader("Simulation labels — comparison only")
        st.write("**Simulated event at this time:**", row["scenario"])
        pipe = row.get("leak_pipe")
        st.write("**Simulated event pipe:**", "None" if str(pipe) == "nan" else str(pipe).removesuffix(".0"))
        st.caption("Labels never enter the detector. A normal label after closure can include hydraulic recovery. Normal replay includes training-period readings; this is not an accuracy evaluation.")
    p, f = st.columns(2)
    with p:
        st.plotly_chart(sensor_figure(replay.frame, index, pressure, "Pressure (metres of head)"), width="stretch")
    with f:
        st.plotly_chart(sensor_figure(replay.frame, index, flow, "Flow (m³/s, signed)"), width="stretch")


render_replay()
