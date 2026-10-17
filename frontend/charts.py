"""Plotly views using ai.topology's graph and original link orientation."""
import plotly.graph_objects as go


def network_figure(graph, row, pressure, flow, suspected):
    fig = go.Figure()
    for start, end, link in graph.edges(data=True):
        a, b = graph.nodes[start], graph.nodes[end]
        pipe = link["pipe_id"]
        sensor = f"F_{pipe}"
        selected = sensor == flow
        pump = link["kind"] == "pump"
        color = "#dc2626" if pipe == suspected else "#0284c7" if selected else "#8b5cf6" if pump else "#94a3b8"
        value = f"{row[sensor]:.6f} m³/s" if sensor in row else "No flow sensor"
        fig.add_trace(go.Scatter(
            x=[a["x"], b["x"]], y=[a["y"], b["y"]], mode="lines",
            line=dict(color=color, width=6 if selected or pipe == suspected else 3, dash="dash" if pump else "solid"),
            name=f'{link["kind"].title()} {pipe}',
            text=f'{link["kind"].title()} {pipe}: {link["start"]} → {link["end"]}<br>Simulated flow: {value}',
            hovertemplate="%{text}<extra></extra>", showlegend=False,
        ))
        fig.add_annotation(x=(a["x"]+b["x"])/2, y=(a["y"]+b["y"])/2,
                           text=f'{"Pump " if pump else ""}{pipe}', showarrow=False,
                           font=dict(size=10, color=color), yshift=10)
    for kind, symbol, color in [("junction", "circle", "#0d9488"), ("tank", "square", "#2563eb"), ("reservoir", "diamond", "#8b5cf6")]:
        nodes = [(n, d) for n, d in graph.nodes(data=True) if d["type"] == kind]
        fig.add_trace(go.Scatter(
            x=[d["x"] for _, d in nodes], y=[d["y"] for _, d in nodes],
            mode="markers+text", name=kind.title(), text=[n for n, _ in nodes], textposition="top center",
            marker=dict(symbol=symbol, color=color, size=[19 if f"P_{n}" == pressure else 13 for n, _ in nodes]),
            hovertext=[f'{kind.title()} {n}<br>Simulated pressure: {row[f"P_{n}"]:.3f} metres of head' if f"P_{n}" in row else f'{kind.title()} {n}<br>No pressure sensor' for n, _ in nodes],
            hovertemplate="%{hovertext}<extra></extra>",
        ))
    fig.update_layout(height=440, margin=dict(l=10,r=10,t=10,b=10), template="plotly_white",
                      xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"), legend=dict(orientation="h"))
    return fig


def sensor_figure(frame, index, sensor, unit):
    past = frame.iloc[:index+1]
    fig = go.Figure(go.Scatter(x=past.timestamp / 3600, y=past[sensor], mode="lines", name=sensor,
                               line=dict(color="#0d9488", width=2)))
    fig.add_vline(x=frame.timestamp.iloc[index]/3600, line_dash="dot", line_color="#64748b")
    fig.update_layout(title=f"Simulated {sensor}", xaxis_title="Elapsed time (hours)", yaxis_title=unit,
                      template="plotly_white", height=280, margin=dict(l=15,r=15,t=45,b=35))
    fig.update_xaxes(range=[frame.timestamp.iloc[0]/3600, frame.timestamp.iloc[-1]/3600])
    return fig
