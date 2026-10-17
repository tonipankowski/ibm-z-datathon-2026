"""Seeded synthetic sensor-error sensitivity suite, not calibrated sensor data."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from simulator.scenarios import ROOT, START, run

BASE_SEED=20261017
OUT=ROOT/'data/sensor_study'


def add_noise(clean, seed):
    rng=np.random.default_rng(seed)
    noisy=clean.copy(deep=True)
    settings={}
    for col in clean.filter(regex='^[PF]_'):
        # Fixed sensor bias and independent sample noise; never use scenario labels.
        sigma=0.10 if col.startswith('P_') else 0.00002
        bias_sigma=0.05 if col.startswith('P_') else 0.00001
        bias=float(rng.normal(0,bias_sigma))
        sample_sigma=np.full(len(clean),sigma) if col.startswith('P_') else np.maximum(sigma,0.005*np.abs(clean[col].to_numpy()))
        noisy[col]=clean[col]+bias+rng.normal(0,sample_sigma)
        settings[col]={'fixed_bias':bias,'bias_sd':bias_sigma,'sample_sd_floor':sigma,'sample_relative_sd':0 if col.startswith('P_') else 0.005}
    return noisy,settings


def main():
    baseline=pd.read_csv(ROOT/'data/generated/normal.csv',keep_default_na=False)
    manifest=[]
    cases=[('normal',None),('demand_spike',None)]+[(kind,pipe) for pipe in ['12','11','31','112'] for kind in ['leak','rupture']]
    for index,(kind,pipe) in enumerate(cases):
        name=kind if pipe is None else f'{kind}_pipe{pipe}'
        dest=OUT/name; dest.mkdir(parents=True,exist_ok=True)
        clean=run(kind,pipe or '12',dest)
        sensors=list(clean.filter(regex='^[PF]_'))
        before=clean.timestamp<START
        np.testing.assert_allclose(clean.loc[before,sensors],baseline.loc[before,sensors],rtol=.002,atol=.002)
        noisy,settings=add_noise(clean,BASE_SEED+index)
        again,_=add_noise(clean,BASE_SEED+index)
        pd.testing.assert_frame_equal(noisy,again)
        pd.testing.assert_frame_equal(clean.drop(columns=sensors),noisy.drop(columns=sensors))
        assert np.isfinite(noisy[sensors].to_numpy()).all()
        noisy.to_csv(dest/f'{kind}.noisy.csv',index=False)
        for variant,df in [('clean',clean),('noisy',noisy)]:
            if kind=='normal':
                df[df.timestamp<3*86400].to_csv(dest/f'train.{variant}.csv',index=False)
                df[df.timestamp>=3*86400].to_csv(dest/f'holdout.{variant}.csv',index=False)
        meta={'case':name,'seed':BASE_SEED+index,'sensor_errors':settings,'assumed_not_calibrated':True,'clean_sha256':hashlib.sha256((dest/f'{kind}.csv').read_bytes()).hexdigest(),'noisy_sha256':hashlib.sha256((dest/f'{kind}.noisy.csv').read_bytes()).hexdigest(),'physics_truth':f'{kind}.truth.csv','noise_applied_after_hydraulics':True}
        (dest/'noise.metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
        manifest.append(meta)
        print(name,'noise reproducibility and labels passed',flush=True)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__': main()
