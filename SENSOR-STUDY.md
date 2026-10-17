# Additional pipes and sensor-error study

Run `python -m simulator.noisy_suite` from this project with its Python environment active. Existing `data/generated/` CSVs are not modified. New outputs are in `data/sensor_study/`.

## Cases and outputs

Ten cases: normal, demand spike, and small-leak/large-opening rupture on each of pipes 12, 11, 31, and 112. Each has a clean CSV, `.noisy.csv`, EPANET input, metadata, and separate `.truth.csv`. All contain 2017 timestamps across seven days at five-minute intervals. Events remain day 3 noon to 6 pm, using the original 5 mm/50 mm orifices. Pipe labels now match each actual split pipe. Flow sensors on affected pipes represent their upstream halves; the downstream halves remain internal to EPANET. This placement differs from intact-network endpoint assumptions used by localization.

Normal has separate `train.clean.csv`, `holdout.clean.csv`, `train.noisy.csv`, and `holdout.noisy.csv`; training covers only the first three days. Do not train on truth files or combine event files into training.

## Assumed sensor error, not manufacturer specifications

Noise is applied only to exported pressure/flow columns, AFTER EPANET solves the physics. Timestamps, scenario labels, pipe labels, and truth remain exact. No hydraulic readings are replaced with fabricated physics.

- Pressure: a fixed Gaussian sensor offset with standard deviation 0.05 m head, plus independent Gaussian sample noise with standard deviation 0.10 m head.
- Flow: fixed Gaussian offset with standard deviation 0.00001 m³/s (0.01 L/s), plus sample noise with standard deviation max(0.5% of absolute clean flow, 0.00002 m³/s = 0.02 L/s).
- Seed: 20261017 plus deterministic case index, recorded in every noise metadata file. Realized fixed offsets are saved per sensor.
- Signed flow is retained without clipping. Small sign changes near zero are possible sensor error. Bias is fixed within each seven-day case; each case uses an independently sampled virtual sensor installation. Consequently comparisons across cases also include calibration differences. The normal training and holdout portions share the same sensor offsets, with different sample errors.

These are illustrative engineering sensitivity assumptions, not validated sensor accuracy or guaranteed realism. Gaussian independent noise omits drift, correlated disturbances, quantization, dropouts, calibration changes over time, and telemetry latency. To claim realistic hardware performance, replace these settings with chosen sensor specifications or measured residuals. This first suite uses one seed per case and one noise level; it does not establish robust performance across random realizations.

## Validation

All ten clean runs passed native EPANET convergence/completeness checks, junction flow conservation, and pre-event agreement against the original normal baseline. Every leak case passed orifice-flow and on/off timing checks. Each noisy export passed exact repeatability for its seed, unchanged-label checks, and finite-value checks. Physical conservation is checked on clean hydraulics, not corrupted sensor readings.

## Updated AI evaluation

The `ai/improve-detection` branch was retrieved via the GitHub connector after command-line Git could not resolve github.com. Its AI files were tested in an isolated workspace; this branch's AI code was not overwritten. Retrieved Git blob IDs:

- train.py: 9d582fe8377ccf9ccad4e44b8612d9e5797eeb14
- detect.py: e32f58ba182227c7b26b6aeefa62873e7c3a5486
- topology.py: 266e5b74518f17da7f2f62eac1a1b7a902ff1101
- __init__.py: d2ac95c9050a84a65dfc9c448469e7c7ffda0fda

Results are in SENSOR-STUDY-EVALUATION.json. Separate clean/noisy models are trained on corresponding normal training data. Normal holdout uses full replay; its denominator includes initial warm-up timestamps. Event metrics cover only the 72 active timestamps. Each event uses a fresh detector, with 19 pre-event calls to initialize features and alert streak; these are event-window checks, not full-run post-event metrics. No thresholds are tuned. The local evaluation uses Mac arm64, not LinuxONE. Event labels and truth are used only for scoring.

### Results of the completed local check

Clean normal holdout: 10/1153 alerting timestamps (0.87%). Noisy normal holdout: 22/1153 (1.91%).

| Case | Clean event alerts /72 | Noisy event alerts /72 | Noisy correct-pipe alerts |
|---|---:|---:|---:|
| Small leak 11 | 0 | 0 | 0 |
| Small leak 112 | 0 | 0 | 0 |
| Small leak 12 | 70 | 12 | 10 |
| Small leak 31 | 70 | 70 | 70 |
| Large opening 11 | 70 | 70 | 70 |
| Large opening 112 | 70 | 70 | 0 |
| Large opening 12 | 70 | 70 | 70 |
| Large opening 31 | 70 | 70 | 70 |
| Demand spike | 70 | 70 | N/A |

Large openings on 11 and 112 were not correctly classified as rupture during alerting timestamps. The noisy pipe-12 small leak's first alert took 4800 seconds (80 minutes); other detected events first alerted after 600 seconds. Demand spike was classified correctly at all 70 alerting timestamps in both variants. These are one-seed results under assumed error, not generalization claims.

After integrating the intended AI branch, reproduce using `python -m simulator.evaluate_sensor_study`. It evaluates whichever AI code is currently checked out; ensure it is Antoni's intended version. It writes the summary and local model files, never edits AI source. The recorded results above used the blob versions listed earlier, not the older AI source still in this simulation branch.
