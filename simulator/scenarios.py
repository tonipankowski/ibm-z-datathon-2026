"""EPANET 2.2 scenario suite. All event parameters are demonstration assumptions."""
from pathlib import Path
import hashlib
import json
import tempfile
import numpy as np
import pandas as pd
import wntr
from wntr.network.controls import Control, ControlAction, SimTimeCondition

ROOT = Path(__file__).resolve().parents[1]
STEP = 300
DURATION = 7*86400
START, END = 3*86400+12*3600, 3*86400+18*3600


def build(kind):
    raw = (ROOT/'data/Net1.inp').read_text()
    fixed = raw.replace(' Type            Bulk         -0.5','; Type            Bulk         -0.5').replace(' Type            Wall         -1.0','; Type            Wall         -1.0')
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'network.inp'; p.write_text(fixed)
        w=wntr.network.WaterNetworkModel(str(p))
    w.options.time.duration=DURATION
    w.options.time.hydraulic_timestep=STEP
    w.options.time.report_timestep=STEP
    w.options.time.pattern_timestep=STEP
    # Resample existing 2-hour demand patterns without changing their values.
    for _, pattern in w.patterns():
        pattern.multipliers=np.repeat(pattern.multipliers, 7200//STEP)
    w.options.quality.parameter='NONE'
    meta={'scenario':kind,'start_s':START,'end_s':END,'pipe_id':None,'assumptions':True}
    if kind in ('leak','rupture'):
        w=wntr.morph.split_pipe(w,'12','12_downstream','event_midpoint')
        node=w.get_node('event_midpoint')
        w.add_junction('event_outlet',base_demand=0,elevation=node.elevation,coordinates=node.coordinates)
        diameter=0.005 if kind=='leak' else 0.05
        coeff=0.62*(np.pi*diameter**2/4)*np.sqrt(2*9.80665)
        w.get_node('event_outlet').emitter_coefficient=coeff
        w.options.hydraulic.emitter_exponent=0.5
        w.add_pipe('event_connection','event_midpoint','event_outlet',length=0.01,diameter=1,roughness=140,initial_status='CLOSED')
        for time,status in [(START,1),(END,0)]:
            w.add_control(f'event_{time}',Control(SimTimeCondition(w,'=',time),ControlAction(w.get_link('event_connection'),'status',status)))
        meta.update(pipe_id='12',hole_diameter_m=diameter,discharge_coefficient=0.62,emitter_coefficient_SI=coeff,
                    representation='midpoint orifice with switched low-resistance connector; rupture is a large opening, not full pipe separation',
                    flow_sensor='F_12 measures upstream half; downstream half retained internally')
    elif kind=='demand_spike':
        for name,node in w.junctions():
            for i,ts in enumerate(node.demand_timeseries_list):
                vals=np.array([ts.at(t)/ts.base_value if ts.base_value else 1 for t in range(0,DURATION+STEP,STEP)], dtype=float)
                times=np.arange(0,DURATION+STEP,STEP)
                vals[(times>=START)&(times<END)]*=1.5
                pname=f'spike_{name}_{i}'; w.add_pattern(pname,vals); ts.pattern_name=pname
        meta.update(demand_multiplier_during_event=1.5,affected_nodes='all demand junctions')
    elif kind!='normal':
        raise ValueError(kind)
    return w,meta


def run(kind):
    w,meta=build(kind)
    out=ROOT/'data/generated'; out.mkdir(parents=True,exist_ok=True)
    wntr.network.write_inpfile(w,str(out/f'{kind}.inp'))
    with tempfile.TemporaryDirectory() as d:
        result=wntr.sim.EpanetSimulator(w).run_sim(file_prefix=str(Path(d)/'run'),version=2.2,convergence_error=True)
    if result.error_code not in (None,0): raise RuntimeError(result.error_code)
    pressure=result.node['pressure']; flow=result.link['flowrate']
    if not np.array_equal(pressure.index,np.arange(0,DURATION+STEP,STEP)): raise RuntimeError('Incomplete results')
    df=pd.DataFrame({'timestamp':pressure.index})
    for node in ['10','11','12','13','21','22','23','31','32']: df[f'P_{node}']=pressure[node].to_numpy()
    for link in ['10','11','12','21','22','31','110','111','112','113','121','122','9']: df[f'F_{link}']=flow[link].to_numpy()
    assert np.isfinite(df.to_numpy()).all()
    active=(df.timestamp>=START)&(df.timestamp<END)&(kind!='normal')
    df['scenario']=np.where(active,kind,'normal')
    df['leak_pipe']=np.where(active & (kind in ('leak','rupture')),'12','')
    df.to_csv(out/f'{kind}.csv',index=False)
    # Separate ground truth; never use actual leakage as an AI input feature.
    truth=pd.DataFrame({'timestamp':df.timestamp,'event_active':active,'leak_flow_m3s':flow['event_connection'].to_numpy() if kind in ('leak','rupture') else 0})
    truth.to_csv(out/f'{kind}.truth.csv',index=False)
    if kind in ('leak','rupture'):
        assert truth.loc[active,'leak_flow_m3s'].min()>0
        assert truth.loc[~active,'leak_flow_m3s'].abs().max()<1e-8
        h=pressure['event_outlet'].to_numpy()[active]
        np.testing.assert_allclose(truth.loc[active,'leak_flow_m3s'],meta['emitter_coefficient_SI']*np.sqrt(h),rtol=0.002,atol=1e-7)
    # Conservation at all junctions, including artificial leak junctions.
    for name,node in w.junctions():
        balance=np.zeros(len(df))
        for link_name,link in w.links():
            if link.end_node_name==name: balance+=flow[link_name].to_numpy()
            if link.start_node_name==name: balance-=flow[link_name].to_numpy()
        np.testing.assert_allclose(balance,result.node['demand'][name],atol=2e-6,rtol=0.002)
    meta.update(engine='EPANET 2.2',wntr_version=wntr.__version__,duration_s=DURATION,step_s=STEP,rows=len(df),
                pressure_unit='metres of pressure head',flow_unit='m3/s, signed',source='simulated, not measured',
                source_sha256=hashlib.sha256((ROOT/'data/Net1.inp').read_bytes()).hexdigest(),
                csv_sha256=hashlib.sha256((out/f'{kind}.csv').read_bytes()).hexdigest(),
                minimum_sensor_pressure_m=float(df.filter(regex='^P_').min().min()))
    (out/f'{kind}.metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(kind,len(df),'rows; physics checks passed')
    return df


def main():
    frames={kind:run(kind) for kind in ['normal','leak','rupture','demand_spike']}
    sensors=list(frames['normal'].filter(regex='^[PF]_').columns)
    for kind in ['leak','rupture','demand_spike']:
        before=frames[kind].timestamp<START
        np.testing.assert_allclose(frames[kind].loc[before,sensors],frames['normal'].loc[before,sensors],rtol=0.002,atol=0.002)
    normal=frames['normal']
    normal[normal.timestamp<3*86400].to_csv(ROOT/'data/generated/normal_train.csv',index=False)
    normal[normal.timestamp>=3*86400].to_csv(ROOT/'data/generated/normal_holdout.csv',index=False)
    print('Pre-event agreement passed; chronological training/holdout CSVs saved.')

if __name__=='__main__': main()
