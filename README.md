# ibm-z-datathon-2026

## Database

AquaWatch stores sensor readings and AI alerts in a SQLite file, `data/aquawatch.db` (gitignored). It uses only Python's built-in `sqlite3` module plus pandas: no server, ORM or Docker, so it runs on IBM LinuxONE (s390x). WAL mode lets the dashboard read while the pipeline is still writing.

### Schema

| Table | Key | Columns |
|---|---|---|
| `runs` | `run_id` | `scenario_file`, `model_path`, `started_at`, `finished_at`, `rows_processed`, `avg_detect_ms`, `platform` (e.g. `s390x`) |
| `readings` | `(run_id, timestamp)` | one `REAL` column per sensor: `P_10` … `P_32`, `F_9` … `F_122` |
| `alerts` | `(run_id, timestamp)` | `is_anomaly`, `severity`, `level`, `suspected_pipe`, `event_type`, `explanation`, `lost_flow_lps`, `pipe_scores` (JSON) |
| `truth` | `(run_id, timestamp)` | `scenario`, `leak_pipe`. For display and evaluation only, never fed to the model |

There is an index on `alerts(run_id, is_anomaly)`.

### Run the pipeline

```bash
python -m backend.run_pipeline --data data/generated/rupture.csv            # as fast as possible
python -m backend.run_pipeline --data data/generated/rupture.csv --speed 5  # live demo: 5 rows/s
```

Options: `--model ai/model.joblib`, `--inp data/Net1.inp`, `--db data/aquawatch.db`. Each run gets a new `run_id`. The pipeline streams the CSV row by row through `LeakDetector.detect()`, times every call, and commits at least every 100 rows or every second.

### Read it from the frontend

```python
from backend import db

run_id = int(db.list_runs()["run_id"].iloc[0])   # newest run
db.get_kpis(run_id)                               # steps, alerts, first alert, avg ms, platform
db.get_readings(run_id, until_ts=t)               # sensor history up to time t (DataFrame)
db.get_alerts(run_id, until_ts=t, only_anomalies=True)  # alert history, pipe_scores as dicts
db.get_latest_alert(run_id, at_ts=t)              # same dict format as detect(), or None
```

Tests: `python -m pytest tests/test_db.py`.