# AquaWatch simulated-data dashboard

Branch: `frontend/aquawatch-dashboard`, based on `simulation/physics-scenarios`.

## Run locally

From the repository root, use Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-frontend.txt
python -m ai.train --data data/sensor_study/normal/train.clean.csv --out ai/model.joblib
python -m streamlit run dashboard.py
```

On Windows, activate with `.venv\Scripts\Activate.ps1` instead. Open
http://localhost:8501. No EPANET installation or regeneration is needed: the
checked-in clean sensor-study CSVs supply normal, small leak on pipe 12,
rupture on pipe 12, and demand spike. See SIMULATOR.md for physical assumptions.
These are simulated readings, not measurements. LinuxONE deployment has not
been validated; run on your local computer first.

Choose the scenario and independent pressure-junction / flow-link sensors in
the sidebar. Play advances by the selected number of five-minute readings per
second; Pause stops playback. Dragging the slider pauses and seeks. At the end,
Play restarts. Charts show only readings up to the current time. Signed flow
uses original INP link orientation; pressure is metres of head, flow is m³/s.
The map is a schematic from `ai.topology.load_network(data/Net1.inp)` and includes
pipes, junctions, tank, reservoir and pump. The event model splits pipe 12;
this intact topology is used by the detector, with the localization limitations
described in SIMULATOR.md.

## Detector integration / handoff to Antoni

`frontend.replay.Replay` calls the existing `ai.detect.LeakDetector` unchanged.
Each browser session owns its detector. Forward seeks process every skipped
reading exactly once using the trailing `2 * window - 1` raw rows. Backward
seeks reuse the previously computed output, preserving the consecutive streak.
Changing scenario or model file creates a new detector. Only timestamp and
the model's sensor columns enter `detect`; scenario and leak_pipe labels are
displayed separately. No truth CSV or event label is a detector input.

The alert panel uses real severity, level, suspected_pipe, event_type,
is_anomaly and explanation. Warm-up and unconfirmed anomalies retain the
detector's wording. Its explanation can say “confidence”; the adjacent warning
clarifies that this is a relative localization weight, not calibrated
probability or proven accuracy. False positives remain visible, including in
the normal scenario. No fabricated or replacement alerts are used on errors.

Train only on the supplied chronological normal training split. Normal replay
also includes training rows, so the dashboard is a demonstration, not an
independent accuracy evaluation. After event closure a normal simulation label
may still correspond to hydraulic recovery. `AQUAWATCH_MODEL` optionally selects
a different trusted local joblib model; do not load untrusted joblib files.

Coordination attempt: GitHub rejected creation of an integration-review issue
with HTTP 403 (integration lacks repository write access). Antoni has **not**
confirmed this contract. Please review the window size, reset behavior, output
fields, and chosen baseline with him before merging.

## Tests

```bash
python -m pip install pytest
python -m pytest tests/test_frontend.py -q
```

Tests compare actual detector outputs with direct chronological calls across
all four scenarios, verify backward/forward seeking and label isolation, map
contents and units, and exercise Streamlit controls and missing-model handling.
Streamlit playback uses its documented fragment API:
https://docs.streamlit.io/develop/api-reference/execution-flow/st.fragment.
