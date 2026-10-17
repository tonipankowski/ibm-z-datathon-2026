"""Run against the AI version checked out in this repository; do not mix model versions."""
from pathlib import Path
import json
import pandas as pd
from ai.train import train
from ai.detect import LeakDetector, replay
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/sensor_study'
rows=[]
for variant in ['clean','noisy']:
    model=BASE/'models'/f'{variant}.joblib'
    train(BASE/f'normal/train.{variant}.csv',model)
    results=replay(BASE/f'normal/holdout.{variant}.csv',model_path=model,inp_path=ROOT/'data/Net1.inp',verbose=False)
    rows.append({'variant':variant,'case':'normal_holdout','rows':len(results),'alerts':sum(r['is_anomaly'] for r in results)})
    print(rows[-1],flush=True)
    for folder in sorted(BASE.iterdir()):
        if not folder.is_dir() or folder.name=='normal':continue
        kind=folder.name.split('_pipe')[0]
        filename=f'{kind}.csv' if variant=='clean' else f'{kind}.noisy.csv'
        df=pd.read_csv(folder/filename)
        detector=LeakDetector(model,ROOT/'data/Net1.inp')
        history=2*detector.window-1
        # Reset each event; preload enough pre-event rows for features and streak.
        start_idx=int(df.index[df.timestamp==302400][0])
        alerts=[]
        for i in range(start_idx-history,start_idx+72):
            res=detector.detect(df.iloc[max(0,i-history+1):i+1])
            if i>=start_idx and res['is_anomaly']:alerts.append(res)
        pipe=folder.name.split('_pipe')[-1] if '_pipe' in folder.name else None
        rows.append({'variant':variant,'case':folder.name,'event_rows':72,'event_alerts':len(alerts),'correct_pipe_alerts':sum(str(r['suspected_pipe'])==pipe for r in alerts) if pipe else None,'correct_type_alerts':sum(r['event_type']==kind for r in alerts),'first_alert_delay_s':float(alerts[0]['timestamp'])-302400 if alerts else None})
        print(rows[-1],flush=True)
        (ROOT/'SENSOR-STUDY-EVALUATION.json').write_text(json.dumps(rows,indent=2)+'\n')
