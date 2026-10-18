from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from frontend.db_view import (
    PIPELINE_COMMAND, runs, timestamps, snapshot, format_time, platform_name,
    alert_status, load_layout, network_figure, sensor_figure,
)
DB_PATH = Path(os.environ.get("AQUAWATCH_DB", str(ROOT / "data/aquawatch.db")))


import streamlit as st
import plotly.graph_objects as go

# Page settings
st.set_page_config(
    page_title="AquaWatch",
    page_icon="💧",
    layout="wide"
)

# Explicit colors also cover a browser with a previously saved dark preference.
st.markdown("""
<style>
:root { color-scheme: light !important; }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #f5f8fc !important; color: #172b45 !important;
}
[data-testid="stHeader"] { background: #f5f8fc !important; }
[data-testid="stSidebar"] { background: #ffffff !important; border-right: 1px solid #e2e9f1; }
[data-testid="stSidebarContent"] { background: #ffffff !important; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1440px; }
[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"],
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p,
[data-testid="stSidebar"] p, label { color: #475569 !important; }
h1, h2, h3 { color: #172b45 !important; letter-spacing: -.035em; }
h2 { font-size: 1.45rem !important; font-weight: 650 !important; padding-top: .3rem !important; }
h3 { font-size: 1.1rem !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: #64748b !important; }
hr { border-color: #e2e9f1 !important; margin: 1.5rem 0 !important; }
[data-testid="stMetric"] {
    background: #ffffff; padding: 20px; border: 1px solid #e2e9f1;
    border-radius: 14px; box-shadow: 0 3px 12px rgba(23,43,69,.035);
    border-top: 3px solid #0891b2;
}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div {
    color: #172b45 !important; font-size: 1.8rem !important; font-weight: 650;
}
[data-testid="stMetricLabel"] p { font-size: .82rem !important; font-weight: 550; }
[data-baseweb="select"] > div, [data-baseweb="input"], input {
    background: #ffffff !important; color: #172b45 !important; border-color: #cbd5e1 !important;
}
[data-baseweb="select"] span, [data-baseweb="select"] input { color: #172b45 !important; }
[data-baseweb="popover"], [role="listbox"], [role="option"] {
    background: #ffffff !important; color: #172b45 !important;
}
[data-testid="stSlider"] { color: #475569 !important; }
[data-testid="stSlider"] [data-testid="stTickBar"] { color: #475569 !important; }
button[kind="secondary"], button[kind="header"] {
    background: #ffffff !important; color: #172b45 !important; border-color: #cbd5e1 !important;
}
button[kind="secondary"] p { color: #172b45 !important; }
button[kind="primary"] { background: #0369a1 !important; border-color: #0369a1 !important; }
button[kind="primary"] p { color: #ffffff !important; }
[data-testid="stAlert"] { background: #eaf4fa !important; color: #173e59 !important; border-radius: 12px; }
[data-testid="stPlotlyChart"] { background: #ffffff; border: 1px solid #e2e9f1; border-radius: 14px; overflow: hidden; }
.aw-hero { display:flex; align-items:center; gap:16px; margin: 0 0 20px; }
.aw-mark { display:grid; place-items:center; width:52px; height:52px; flex-shrink:0;
    border-radius:16px; background:linear-gradient(145deg,#0284c7,#0891b2); color:white; font-size:28px; }
.aw-eyebrow { color:#0369a1; font-size:.7rem; text-transform:uppercase; letter-spacing:.14em; font-weight:700; }
.aw-title { color:#172b45; font-size:2.25rem; line-height:1.2; font-weight:750; letter-spacing:-.05em; }
.aw-subtitle { color:#64748b; font-size:.95rem; margin-top:4px; }
.aw-note { background:#eaf3f9; border:1px solid #d5e7f2; color:#436176; border-radius:10px;
    padding:12px 16px; font-size:.8rem; line-height:1.6; }
@media(max-width:640px) { .aw-title {font-size:1.8rem;} .block-container {padding-top:1rem;} }
</style>
<div class="aw-hero">
  <div class="aw-mark">≈</div>
  <div><div class="aw-eyebrow">Water network intelligence</div>
    <div class="aw-title">AquaWatch</div>
    <div class="aw-subtitle">Monitor pressure. Detect anomalies. Prioritize inspection.</div>
  </div>
</div>
<div class="aw-note"><strong>SIMULATION WORKSPACE</strong> &nbsp;·&nbsp; EPA EPANET Net1 &nbsp;·&nbsp; IBM Z Datathon 2026<br>
Pressure in metres of head · Flow in m³/s. Detector confidence is a heuristic, not proven accuracy.</div>
""", unsafe_allow_html=True)

# Sidebar navigation (original visual structure retained).
with st.sidebar:
    st.title("💧 AquaWatch")
    st.caption("Network Operations Center")
    st.divider()
    st.markdown("### Navigation")
    st.markdown("📊 **Dashboard**")
    st.divider()
    live = st.toggle("Live mode · refresh every 2 seconds", key="live", value=False)
    replay_speed = st.select_slider(
        "Replay speed", options=[1, 6, 12, 60, 120], value=12,
        format_func=lambda value: f"{value} readings / second",
        key="replay_speed", disabled=live,
    )
    st.caption("At 5-minute sampling, 12 readings/second replays one simulated hour per second.")
    if st.button("Refresh database"):
        st.cache_data.clear()
    st.markdown("### System Information")
    st.write("Network: EPANET Net1")
    st.write("Mode: Simulated data")
    st.write("Backend: SQLite pipeline results")




def no_data(message):
    st.info(message)
    st.code(PIPELINE_COMMAND, language="bash")


@st.fragment(run_every=2 if live else None)
def wait_for_runs():
    if DB_PATH.exists():
        try:
            if not runs(str(DB_PATH)).empty:
                st.rerun()
        except Exception:
            pass
    no_data("No pipeline runs are available. Create simulated readings and detector results with:")


if not DB_PATH.exists():
    wait_for_runs()
    st.stop()
try:
    available_runs = runs(str(DB_PATH))
except Exception as exc:
    st.error(f"Cannot read the AquaWatch database: {exc}")
    st.code(PIPELINE_COMMAND, language="bash")
    st.stop()
if available_runs.empty:
    wait_for_runs()
    st.stop()

run_ids = available_runs.run_id.astype(int).tolist()
run_labels = {int(row.run_id): f"{str(row.scenario_file).replace(chr(92), '/').split('/')[-1]} · run {row.run_id}"
              for row in available_runs.itertuples()}
with st.sidebar:
    if st.session_state.get("run_id") not in run_ids:
        st.session_state.run_id = run_ids[0]
    run_id = st.selectbox("Recorded scenario run", run_ids, format_func=run_labels.get, key="run_id")
    st.divider()
    st.caption("IBM Z Datathon 2026")

if st.session_state.get("active_run") != run_id:
    st.session_state.active_run = run_id
    st.session_state.playing = False
    st.session_state.pop("selected_time", None)
st.session_state.setdefault("playing", False)
if live:
    st.session_state.playing = False


def toggle_playback():
    st.session_state.playing = not st.session_state.playing
    st.session_state.control_changed = True
    if st.session_state.playing and st.session_state.get("at_end"):
        st.session_state.pop("selected_time", None)


def seek():
    st.session_state.playing = False
    st.session_state.seeked = True


try:
    layout = load_layout(ROOT)
except Exception as exc:
    st.error(f"Cannot load the network map: {exc}")
    st.stop()


@st.fragment(run_every=2 if live else (1 if st.session_state.playing else None))
def dashboard():
    if st.session_state.pop("seeked", False) or st.session_state.pop("control_changed", False):
        st.rerun()
    try:
        if live and runs(str(DB_PATH)).run_id.astype(int).tolist() != run_ids:
            st.rerun()
        times = timestamps(run_id, str(DB_PATH))
        if not times:
            no_data("This run has not written any readings yet. Live mode will wait for the pipeline.")
            return
        selected = st.session_state.get("selected_time", times[0])
        if selected not in times:
            selected = times[0]
        if live:
            selected = times[-1]
        elif st.session_state.playing:
            selected = times[min(times.index(selected) + replay_speed, len(times)-1)]
        st.session_state.selected_time = selected
        st.session_state.at_end = selected == times[-1]
        if st.session_state.playing and st.session_state.at_end:
            st.session_state.playing = False
            st.rerun()

        # Reserve the original overview and scenario sections; fill them after
        # reading the selected timestamp so every card uses the same cutoff.
        st.divider()
        overview = st.container()
        st.divider()
        st.header("Scenario Testing")
        st.caption(f"Recorded simulation: {run_labels[run_id]}")
        st.button("⏸ Pause" if st.session_state.playing else "▶ Play replay", key="play_pause",
                  on_click=toggle_playback, disabled=live, type="primary")
        if len(times) == 1:
            until_ts = times[0]
            st.caption(f"One reading available: {format_time(until_ts)}")
        else:
            until_ts = st.select_slider("Replay timestamp", options=times, key="selected_time",
                                        format_func=format_time, on_change=seek, disabled=live)
        st.caption(f"Showing readings and predictions through {format_time(until_ts)} only.")
        readings, alerts, anomalies, latest, kpis = snapshot(run_id, until_ts, str(DB_PATH))
        with overview:
            st.header("Network Overview")
            cols = st.columns(5)
            for col, label, value in zip(cols,
                ["Total steps", "Alert count", "First alert time", "Avg detect ms", "Platform"],
                [kpis["total_steps"], kpis["alert_count"], format_time(kpis["first_alert_time"]),
                 "—" if kpis["avg_detect_ms"] is None else f"{kpis['avg_detect_ms']:.2f}", platform_name(kpis["platform"])]):
                col.metric(label, value)
            st.caption("Counts are through the selected time. Average latency is available only at the end of a completed run.")
        if latest is None:
            st.info("No detector result is available at this timestamp yet.")
        else:
            headline, event_label, display_name = alert_status(latest)
            display = getattr(st, display_name)
            display(f"{headline} · severity {latest['severity']}/100")

        st.divider()
        st.header("Water Distribution Network")
        st.caption("EPA EPANET Net1 — Interactive Network Map")
        st.plotly_chart(network_figure(layout, latest), width="stretch", key="network")
        st.caption("🟢 Junctions | 🔵 Reservoir | 🟣 Tank | 🟠 Pump. Pipe color: confirmed alert level (green without one); opacity: relative localization weight; thick pipe: suspected pipe. Gray means no result.")

        for pressure, heading, prefix in [(True, "Pressure Monitoring", "P_"), (False, "Flow Monitoring", "F_")]:
            st.divider()
            st.header(heading)
            sensors = [c for c in readings if c.startswith(prefix) and readings[c].notna().any()]
            if not sensors:
                st.info("No sensor readings are available at this time.")
                continue
            sensor = st.selectbox("Pressure sensor" if pressure else "Flow sensor", sensors,
                                  index=sensors.index("P_22") if pressure and "P_22" in sensors else 0,
                                  key="pressure_sensor" if pressure else "flow_sensor")
            st.caption("Simulated readings stored by the pipeline. Red shading marks detector-active alerts, not simulation truth.")
            st.plotly_chart(sensor_figure(readings, anomalies, sensor, pressure), width="stretch", key=heading)

        st.divider()
        st.header("Network Alerts")
        st.caption("Actual stored detector output from this simulated run")
        if latest is None:
            st.info("Waiting for a detector result.")
        else:
            display(headline)
            cols = st.columns(3)
            cols[0].metric("Suspected pipe", latest["suspected_pipe"] or "None")
            cols[1].metric("Severity", f"{latest['severity']}/100")
            cols[2].metric("Event type", event_label)
            st.write(latest["explanation"])
            st.caption(f"Detector timestamp: {format_time(latest['timestamp'])}. Predictions can contain false alarms; localization weights are not calibrated accuracy.")
        st.caption("AquaWatch | IBM Z Datathon | Simulated data")
    except Exception as exc:
        st.session_state.playing = False
        st.error(f"Cannot read this run: {exc}. No substitute readings or alerts are shown.")


dashboard()
