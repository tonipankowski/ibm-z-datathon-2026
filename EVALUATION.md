# Local AI evaluation — 17 October 2026

Antoni's AI code from commit 60952f1 was used unchanged on Mac arm64. Training used normal_train.csv: 864 rows, 855 usable after warm-up, 88 features, window 10. A fresh model was trained locally; no LinuxONE model was downloaded.

| Dataset | Replayed rows | Alerting timestamps | Alerts during 72 event timestamps | Correct pipe 12 during event |
|---|---:|---:|---:|---:|
| Normal holdout | 1153 | 225 | Not applicable | Not applicable |
| Small leak | 2017 | 349 | 0 | 0 |
| Large-opening rupture | 2017 | 340 | 31 | 0 |
| Demand spike | 2017 | 309 | 12 | Not applicable |

Normal holdout raw alert fraction is 225/1153 = 19.5%, including initial warm-up rows in the denominator. Small leak was missed during its active interval. Large-opening first alert was at 303000 seconds (10 minutes after onset), classified as leak on pipe 113, not true pipe 12. Demand spike first alert was also at 303000 seconds, incorrectly classified as a red leak on pipe 31.

Totals on event files include pre-event and post-event timestamps and must not be read as event recall. Those files include the first three days used for training, so their overall totals are not independent test metrics. Post-event recovery may retain hydraulic effects after labels return to normal. Normal holdout repeats prescribed demand patterns and is not an independent measured dataset.

LinuxONE screenshots confirmed matching training summaries, 225 normal holdout alerts, and 349 total small-leak alerts. Detailed event counts and rupture/demand-spike results above were computed locally only.

These results establish executable integration, not detection accuracy or physical validation. See SIMULATOR.md for assumptions, physics checks, topology/sensor-placement limitations, and reproduction commands. Raw summary is Evaluation-results.json. No AI code was changed or tuned to these results.

## Reproduce

```bash
python -m ai.train --data data/generated/normal_train.csv --out ai/model_7day.joblib
python -m ai.detect --data data/generated/normal_holdout.csv --model ai/model_7day.joblib --inp data/Net1.inp
python -m ai.detect --data data/generated/leak.csv --model ai/model_7day.joblib --inp data/Net1.inp
python -m ai.detect --data data/generated/rupture.csv --model ai/model_7day.joblib --inp data/Net1.inp
python -m ai.detect --data data/generated/demand_spike.csv --model ai/model_7day.joblib --inp data/Net1.inp
```

Train only on normal_train.csv. Truth CSVs are evaluation-only, not model features. Generated INP files include the exact simulated networks; ai.detect uses the original topology for candidate pipe IDs as documented.
