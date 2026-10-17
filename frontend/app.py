
import streamlit as st
import plotly.graph_objects as go

# Page settings
st.set_page_config(
    page_title="AquaWatch",
    page_icon="💧",
    layout="wide"
)

# Dashboard title

# Professional dashboard header
st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

h1, h2, h3 {
    letter-spacing: -0.5px;
}

div[data-testid="stMetric"] {
    background-color: #162238;
    padding: 18px;
    border: 1px solid #293A54;
    border-radius: 12px;
}

div[data-testid="stMetricLabel"] {
    color: #94A3B8;
}

div[data-testid="stMetricValue"] {
    color: #F8FAFC;
}
</style>
""", unsafe_allow_html=True)

st.title("💧 AquaWatch")
st.markdown(
    "**Intelligent Water Infrastructure Monitoring**"
)
st.caption(
    "IBM Z Datathon 2026  •  EPA EPANET Net1  •  Prototype Dashboard"
)


# Sidebar navigation
with st.sidebar:
    st.title("💧 AquaWatch")
    st.caption("Network Operations Center")

    st.divider()

    st.markdown("### Navigation")

    st.markdown("📊 **Dashboard**")


    st.divider()

    st.markdown("### System Information")
    st.write("Network: EPANET Net1")
    st.write("Mode: Demonstration")
    st.write("Backend: Not connected")

    st.divider()
    st.caption("IBM Z Datathon 2026")

st.divider()


# Dashboard metrics
st.header("Network Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Junctions", "9")

with col2:
    st.metric("Pipes", "12")

with col3:
    st.metric("Network Components", "11")

with col4:
    active_alerts = (
        1 if st.session_state.get(
            "active_scenario", "Normal Operation"
        ) == "Simulated Leak" else 0
    )
    st.metric("Demo Alerts", active_alerts)


st.divider()

# Scenario selection
st.header("Scenario Testing")

scenario = st.selectbox(
    "Choose a water network scenario",
    ["Normal Operation", "Simulated Leak"]
)


# Remember the scenario after clicking the button
if "active_scenario" not in st.session_state:
    st.session_state.active_scenario = "Normal Operation"

if st.button("▶ Run Scenario", type="primary"):
    st.session_state.active_scenario = scenario

if st.session_state.active_scenario == "Simulated Leak":
    st.error(
        "⚠️ DEMO ALERT: Possible leak on pipe 12 → 22"
    )
    st.warning(
        "Illustrative scenario — not a live leak prediction."
    )
else:
    st.success("Normal Operation — No demo alerts")


st.divider()

# Network map placeholder

import plotly.graph_objects as go

st.header("Water Distribution Network")
st.caption("EPA EPANET Net1 — Interactive Network Map")

# Junction, reservoir, and tank coordinates
nodes = {
    "9": (10, 70),
    "10": (20, 70),
    "11": (40, 70),
    "12": (60, 70),
    "13": (80, 70),
    "21": (40, 40),
    "22": (60, 40),
    "23": (80, 40),
    "31": (40, 10),
    "32": (60, 10),
    "2": (90, 70)
}

# EPANET Net1 pipes
pipes = [
    ("10", "11"), ("11", "12"), ("12", "13"),
    ("21", "22"), ("22", "23"), ("31", "32"),
    ("2", "12"), ("11", "21"), ("12", "22"),
    ("13", "23"), ("21", "31"), ("22", "32")
]

fig = go.Figure()

# Draw all 12 pipes

# Draw pipes and highlight the demo leak
leak_active = (
    st.session_state.active_scenario == "Simulated Leak"
)

for start, end in pipes:
    x1, y1 = nodes[start]
    x2, y2 = nodes[end]

    is_leak_pipe = (
        leak_active and start == "12" and end == "22"
    )

    pipe_color = "#ef4444" if is_leak_pipe else "#38bdf8"
    pipe_width = 9 if is_leak_pipe else 5

    fig.add_trace(go.Scatter(
        x=[x1, x2],
        y=[y1, y2],
        mode="lines",
        line=dict(color=pipe_color, width=pipe_width),
        hovertemplate=(
            "DEMO LEAK: Pipe 12 → 22<extra></extra>"
            if is_leak_pipe
            else f"Pipe {start} → {end}<extra></extra>"
        ),
        showlegend=False
    ))


# Draw pump connection
fig.add_trace(go.Scatter(
    x=[10, 20],
    y=[70, 70],
    mode="lines",
    line=dict(color="#f59e0b", width=5, dash="dash"),
    name="Pump",
    hovertemplate="Pump: Reservoir 9 → Junction 10<extra></extra>"
))

# Draw junctions
junctions = [
    "10", "11", "12", "13", "21",
    "22", "23", "31", "32"
]

fig.add_trace(go.Scatter(
    x=[nodes[n][0] for n in junctions],
    y=[nodes[n][1] for n in junctions],
    mode="markers+text",
    marker=dict(
        size=22,
        color="#22c55e",
        line=dict(color="white", width=2)
    ),
    text=junctions,
    textposition="top center",
    name="Junctions",
    hovertemplate="Junction %{text}<extra></extra>"
))

# Reservoir
fig.add_trace(go.Scatter(
    x=[10], y=[70],
    mode="markers+text",
    marker=dict(size=28, color="#3b82f6", symbol="diamond"),
    text=["Reservoir 9"],
    textposition="top center",
    name="Reservoir",
    hovertemplate="Reservoir 9<extra></extra>"
))

# Tank
fig.add_trace(go.Scatter(
    x=[90], y=[70],
    mode="markers+text",
    marker=dict(size=28, color="#a855f7", symbol="square"),
    text=["Tank 2"],
    textposition="top center",
    name="Tank",
    hovertemplate="Tank 2<extra></extra>"
))

# Map appearance

fig.update_layout(
    height=460,
    paper_bgcolor="#0f172a",
    plot_bgcolor="#0f172a",
    font=dict(color="white"),
    margin=dict(l=20, r=20, t=30, b=65),
    xaxis=dict(visible=False, range=[0, 105]),
    yaxis=dict(
        visible=False,
        range=[0, 90],
        scaleanchor="x",
        scaleratio=1
    ),
    legend=dict(
        orientation="h",
        x=0.5,
        xanchor="center",
        y=-0.12,
        yanchor="top",
        bgcolor="rgba(15,23,42,0.95)",
        font=dict(color="white", size=13)
    )
)


st.plotly_chart(fig, use_container_width=True)

st.caption(
    "🟢 Junctions | 🔵 Reservoir | "
    "🟣 Tank | 🟠 Pump"
)


# Pressure Monitoring Chart
st.divider()
st.header("Pressure Monitoring")
st.caption("Illustrative demo data — not live EPANET results")

hours = [0, 4, 8, 12, 16, 20, 24]

normal_pressure = [58, 57, 59, 56, 57, 58, 58]
leak_pressure = [58, 56, 48, 39, 35, 37, 36]

pressure_fig = go.Figure()

pressure_fig.add_trace(go.Scatter(
    x=hours,
    y=normal_pressure,
    mode="lines+markers",
    name="Normal Pressure",
    line=dict(color="#22c55e", width=3)
))

if st.session_state.active_scenario == "Simulated Leak":
    pressure_fig.add_trace(go.Scatter(
        x=hours,
        y=leak_pressure,
        mode="lines+markers",
        name="Simulated Leak",
        line=dict(color="#ef4444", width=3)
    ))

pressure_fig.update_layout(
    title="Pressure at Junction 22 (Demo)",
    xaxis_title="Time (hours)",
    yaxis_title="Illustrative Pressure (psi)",
    height=380,
    legend=dict(orientation="h", y=1.15)
)

st.plotly_chart(pressure_fig, use_container_width=True)


# Network Alert Panel
st.divider()
st.header("Network Alerts")
st.caption("Illustrative alerts for the hackathon demonstration")

if st.session_state.active_scenario == "Simulated Leak":

    st.error("🚨 HIGH SEVERITY — Potential Pipe Leak")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Affected Pipe", "12 → 22")

    with col2:
        st.metric("Affected Junction", "22")

    with col3:
        st.metric("Demo Pressure Drop", "23 psi")

    st.warning(
        "Recommended Action: Inspect pipe 12 → 22 "
        "and verify pressure readings."
    )

    st.info(
        "Status: Investigation Suggested | "
        "Source: Illustrative Demo Scenario"
    )

else:
    st.success("✅ No Active Demo Alerts")
    st.info(
        "Network operating under the normal demonstration scenario."
    )


# Footer
st.caption("AquaWatch | IBM Z Datathon")
