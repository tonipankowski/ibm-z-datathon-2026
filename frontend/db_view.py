"""Read-only dashboard data and Plotly views; no detector imports or SQL."""
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend import db

PIPELINE_COMMAND = "python -m backend.run_pipeline --data data/generated/rupture.csv"


@st.cache_data(ttl=1, max_entries=8, show_spinner=False)
def runs(path):
    return db.list_runs(db_path=path).sort_values("run_id", ascending=False)


@st.cache_data(ttl=1, max_entries=16, show_spinner=False)
def timestamps(run_id, path):
    # This unbounded read is used only to discover slider options, never chart data.
    return db.get_readings(run_id, db_path=path)["timestamp"].tolist()


@st.cache_data(ttl=1, max_entries=32, show_spinner=False)
def snapshot(run_id, until_ts, path):
    readings = db.get_readings(run_id, until_ts=until_ts, db_path=path)
    alerts = db.get_alerts(run_id, until_ts=until_ts, db_path=path)
    anomalies = db.get_alerts(run_id, until_ts=until_ts, only_anomalies=True, db_path=path)
    latest = db.get_latest_alert(run_id, at_ts=until_ts, db_path=path)
    kpis = db.get_kpis(run_id, db_path=path)
    # get_kpis has no until_ts parameter. Bound its aggregate cards with the
    # allowed alert helper; latency is only stored as a final run-wide average.
    complete = len(alerts) == kpis["total_steps"] and kpis["finished_at"] is not None
    kpis = dict(kpis, total_steps=len(alerts), alert_count=len(anomalies),
                first_alert_time=None if anomalies.empty else anomalies.timestamp.iloc[0],
                avg_detect_ms=kpis["avg_detect_ms"] if complete else None)
    return readings, alerts, anomalies, latest, kpis


def format_time(value):
    if value is None or pd.isna(value):
        return "—"
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (int, float)):
        return f"{value:g} s"
    return str(value)


def platform_name(value):
    return "IBM LinuxONE (s390x)" if value == "s390x" else str(value)


def load_layout(root):
    # Exported using ai.topology.load_network. The checked source hash prevents
    # a changed INP from silently displaying an obsolete layout. This avoids a
    # networkx dependency in the three-package dashboard runtime.
    layout = json.loads((root / "frontend/net1_layout.json").read_text())
    if layout["source_sha256"] != sha256((root / "data/Net1.inp").read_bytes().replace(b"\r\n", b"\n")).hexdigest():
        raise ValueError("Net1.inp changed; regenerate frontend/net1_layout.json with frontend.export_layout.")
    return layout


def network_figure(layout, alert):
    fig = go.Figure()
    nodes = layout["nodes"]
    scores = alert["pipe_scores"] if alert else {}
    level = alert["level"] if alert else None
    suspected = str(alert["suspected_pipe"]) if alert and alert["suspected_pipe"] is not None else None
    color = {"green": "#22c55e", "yellow": "#eab308", "red": "#ef4444"}.get(level, "#64748b")
    peak = max(scores.values(), default=0)
    for link in layout["links"]:
        a, b = nodes[link["start"]], nodes[link["end"]]
        pipe = link["pipe_id"]
        score = scores.get(pipe)
        pump = link["kind"] == "pump"
        highlight = pipe == suspected
        fig.add_trace(go.Scatter(
            x=[a["x"], b["x"]], y=[a["y"], b["y"]], mode="lines",
            line=dict(color="#f59e0b" if pump else color,
                      width=9 if highlight else 5, dash="dash" if pump else "solid"),
            opacity=1 if pump or highlight else (.3 + .7 * score / peak if score is not None and peak else .3),
            name="Pump" if pump else f"Pipe {pipe}", showlegend=pump,
            hovertemplate=(f"{link['kind'].title()} {pipe}: {link['start']} → {link['end']}"
                           + (f"<br>Localization weight: {score:.3f}" if score is not None else "<br>No localization score")
                           + ("<br>Detector-suspected pipe" if highlight else "") + "<extra></extra>"),
        ))
    for kind, name, symbol, node_color, size in [
        ("junction", "Junctions", "circle", "#22c55e", 22),
        ("reservoir", "Reservoir", "diamond", "#3b82f6", 28),
        ("tank", "Tank", "square", "#a855f7", 28),
    ]:
        ids = [n for n, d in nodes.items() if d["type"] == kind]
        fig.add_trace(go.Scatter(x=[nodes[n]["x"] for n in ids], y=[nodes[n]["y"] for n in ids],
            mode="markers+text", marker=dict(size=size, color=node_color, symbol=symbol, line=dict(color="white", width=2)),
            text=[n if kind == "junction" else f"{name} {n}" for n in ids], textposition="top center",
            name=name, hovertemplate=f"{name} %{{text}}<extra></extra>"))
    fig.update_layout(height=460, paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", font=dict(color="white"),
        margin=dict(l=20, r=20, t=30, b=65), xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", x=.5, xanchor="center", y=-.12, yanchor="top",
                    bgcolor="rgba(15,23,42,0.95)", font=dict(color="white", size=13)))
    return fig


def anomaly_intervals(readings, anomalies):
    """Shade only alert-active samples, ending at the next known reading.

    This does not bridge separated alarms or shade beyond the selected timestamp.
    """
    active = set(anomalies.timestamp)
    times = readings.timestamp.tolist()
    intervals = []
    start = None
    for i, timestamp in enumerate(times):
        if timestamp in active and start is None:
            start = timestamp
        if start is not None and (timestamp not in active or i == len(times)-1):
            intervals.append((start, timestamp))
            start = None
    return intervals


def sensor_figure(readings, anomalies, sensor, pressure):
    unit = "Pressure (metres of head)" if pressure else "Flow (m³/s, signed)"
    fig = go.Figure(go.Scatter(x=readings.timestamp, y=readings[sensor], mode="lines",
                               name=sensor, line=dict(color="#22c55e" if pressure else "#38bdf8", width=3)))
    for start, end in anomaly_intervals(readings, anomalies):
        if start == end:
            fig.add_vline(x=start, line_color="#ef4444", line_width=2)
        else:
            fig.add_vrect(x0=start, x1=end, fillcolor="#ef4444", opacity=.15, line_width=0)
    fig.update_layout(title=f"Simulated {'pressure' if pressure else 'flow'} · {sensor}",
        xaxis_title="Elapsed time (seconds)", yaxis_title=unit, height=380,
        legend=dict(orientation="h", y=1.15))
    return fig
