"""Interactive plots of computed sensor data, with event intervals shaded."""
from pathlib import Path
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
ROOT=Path(__file__).resolve().parents[1]
fig=make_subplots(rows=2,cols=1,shared_xaxes=True,subplot_titles=['Pressure at junction 13','Flow into pipe 12 (upstream sensor)'])
for kind in ['normal','leak','rupture','demand_spike']:
    df=pd.read_csv(ROOT/f'data/generated/{kind}.csv')
    for row,col in [(1,'P_13'),(2,'F_12')]:
        fig.add_trace(go.Scatter(x=df.timestamp/86400,y=df[col],name=kind,legendgroup=kind,showlegend=row==1),row=row,col=1)
fig.add_vrect(x0=3.5,x1=3.75,fillcolor='orange',opacity=.15,line_width=0,row='all',col=1)
fig.update_yaxes(title_text='Pressure head (m)',row=1,col=1)
fig.update_yaxes(title_text='Signed flow (m³/s)',row=2,col=1)
fig.update_xaxes(title_text='Elapsed days',row=2,col=1)
fig.update_layout(title='AquaWatch — EPANET simulated readings (not field measurements)',template='plotly_white',height=780)
fig.write_html(ROOT/'Simulation-results.html',include_plotlyjs=True)
