# AquaWatch Role 1 simulator

Open this folder in VS Code. The Python environment is `.venv/bin/python`.

Run in VS Code's Terminal → New Terminal:

```bash
source .venv/bin/activate
python -m simulator.scenarios
python -m simulator.plot_results
```

Open `Simulation-results.html` in a browser to zoom, hover and toggle scenarios. Orange shading is the event interval. Source is `simulator/scenarios.py`; generated CSVs, INP models, metadata and separate truth files are in `data/generated/`.

## Assumptions and calculations

Uses the team's Net1 model from commit 60952f1, including pipe 110. The two invalid Type reaction entries are commented out in memory only. All original hydraulic parameters are retained except explicitly configured time steps and scenarios. Water quality calculation is disabled; this is hydraulic work.

Each run lasts seven days, sampled every 300 seconds: 2017 rows and 22 sensor columns. Existing two-hour demand-pattern values are expanded to five-minute intervals without changing their values. This repeats the same prescribed daily pattern; it is not seven days of independent real-world demand. No synthetic noise or measured data are claimed.

Normal training CSV covers days 0–3 exclusive (864 rows); normal holdout covers day 3 onward (1153 rows). All runs start from the same initial tank state, without a separate equilibration phase. The chronological holdout is not independent physical validation.

Demonstration events occur from 302400 seconds (day 3 at noon) inclusive to 324000 seconds (6 pm) exclusive:

- Small leak: 5 mm circular opening at midpoint of pipe 12.
- Rupture: 50 mm opening at the same midpoint; represents a large orifice, NOT complete severance.
- Demand spike: 1.5 times patterned consumption at all demand junctions.

Leak discharge follows Q = Cd × A × sqrt(2gh), with Cd=0.62, A=pi*d²/4, g=9.80665 m/s². A short 0.01 m long, 1 m diameter connector switches the outlet on/off using native EPANET controls. This connector is an idealized numerical device. The pipe is split into two halves, retaining F_12 as the upstream-half sensor. Its downstream end in the internal event model differs from the intact topology; localization remains to be evaluated accordingly.

EPANET demand-driven hydraulics is retained. Do not extrapolate severe depressurization results without a pressure-dependent demand model. Sensor-pressure minima are recorded in metadata. Pressure is metres of head; flow is signed m³/s in original link orientation. Timestamps are elapsed seconds.

## Validation and limitations

Generation checks finite and complete results, flow conservation at every junction, leak discharge against the orifice equation, zero discharge outside the event, and agreement with the baseline before events. All four runs passed on Mac arm64 with WNTR 1.5.0 / EPANET 2.2.

Scenario labels denote whether the event is active. After closure, tank levels/flows may still be recovering; a `normal` label after the event does not imply instantaneous return to the original baseline. Train using only `normal_train.csv`, not the normal-labelled portions of event CSVs. `.truth.csv` files are evaluation-only and must not be model inputs.

AI files are unchanged. The small earlier baseline caused one false alert; no accuracy claim is made. Longer datasets still require detector evaluation and parameter review. Physical validation requires measured pressure/flow, calibrated demands and asset parameters. LinuxONE has run AI training/replay on the earlier CSV; these new scenario runs were performed only on the Mac. Native EPANET on LinuxONE remains untested.

Install elsewhere on a supported platform with `python -m pip install -r requirements-simulator.txt`. No dashboard dependency is required for generation or standalone HTML charts.

References: https://usepa.github.io/WNTR/controls.html and https://usepa.github.io/WNTR/hydraulics.html
