# AquaWatch SQLite dashboard

Run from the repository root with Python 3.12:

```bash
python -m pip install -r frontend/requirements.txt
streamlit run frontend/app.py
```

The frontend runtime uses only Streamlit, Plotly, pandas and the Python standard
library. It reads `data/aquawatch.db` through the five public `backend.db` reader
functions. It does not import the detector or scikit-learn. Set `AQUAWATCH_DB` to
an alternate database path for testing if needed.

Create a run separately in the team's configured detector environment:

```bash
python -m backend.run_pipeline --data data/generated/rupture.csv
# For a live demonstration (two new readings per second):
python -m backend.run_pipeline --data data/generated/rupture.csv --speed 2
```

The pipeline requires its trained model and existing AI dependencies; these are
not frontend dependencies. A missing or empty DB shows the pipeline command.
Choose a recorded run in the sidebar (newest first). Replay follows actual
timestamps. Live mode follows the latest timestamp of the selected run every
two seconds. Replay defaults to 12 readings per second (one simulated hour per
second for five-minute samples); the sidebar speed control offers 1, 6, 12, 60,
or 120 readings per second. Live mode follows incoming data instead of this speed.
Live mode also waits when the pipeline has not yet flushed its first readings.
Newly created runs appear in the sidebar; selection stays on the user's run.
Database reads have a one-second cache TTL and bounded cache size.

All displayed readings, alerts, shading and count cards respect the selected
timestamp. `get_kpis()` has no cutoff argument, so the frontend uses bounded
`get_alerts()` results for the time-specific counts and first alert time. The
backend stores only a final run-wide average latency: the frontend shows it
only at the end of a completed run, and shows a dash earlier or while writing.
Platform comes from the stored run, with `s390x` displayed as
`IBM LinuxONE (s390x)`. It is never inferred from the dashboard host.

Map geometry in `net1_layout.json` was exported using
`ai.topology.load_network("data/Net1.inp")`. Its source hash is checked at runtime.
This preserves the actual pipe IDs and endpoints (including pipe 12: 12→13),
tank and pump, without requiring networkx on the dashboard. If Net1 changes,
regenerate in the existing backend environment with:

```bash
python -m frontend.export_layout
```

The existing dark cards, header, sidebar and full-width map/chart layout are
retained. Pipe hue shows the latest detector level, opacity shows relative
pipe score, and thickness highlights the suspected pipe. Charts use simulated
pressure in metres of head and signed flow in m³/s. Red shading represents
stored detector anomalies, not event ground truth. False alarms and confidence
wording are shown honestly; no fabricated alerts or sensor values are used.
