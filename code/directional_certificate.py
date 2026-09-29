"""Frozen v3 row maps, exported for anisotropic uncertainty calculations.

No perception labels, coverage results, or test data enter this module. All five
listed constraint families and all 80 steps/followers enter the row maximum.
"""
import csv
import json
import math
from pathlib import Path
import numpy as np
from certify import CONFIG, GRADES, CURVES, truck, net_brake, speed_limit
from certify_common_input import (H, N, maps, LEADER_L2_BUDGET,
                                  W_L2_BUDGET, EPS_STACK_FACTOR)

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'certificates';OUT.mkdir(parents=True,exist_ok=True)
FIELDS=('spacing','stopping','speed_envelope','brake_input','terminal')
SELECTED=[
 dict(name='light_brake',kind='homogeneous',load='L1_light',grade='G1_level',curvature='K1_straight',headway_s=2.0,age_s=.2),
 dict(name='light_stopping',kind='homogeneous',load='L1_light',grade='G1_level',curvature='K1_straight',headway_s=1.0,age_s=.4),
 dict(name='hetero_brake',kind='heterogeneous',load='L3_loaded',grade='G2_downhill',curvature='K2_moderate',headway_s=3.0,age_s=.2),
]

def row_maps(kind,load,grade_name,curve_name,h,age):
    """Yield exact frozen leader/perception/dynamics rows and remaining M_j."""
    m=maps(kind,load,h);seq=m['sequence'];v=CONFIG['speed_mps']
    grade=GRADES[grade_name];curvature=CURVES[curve_name]
    for i in (1,2,3):
        for k in range(H):
            name=seq[k][i];pred=seq[k][i-1];p=truck(name)
            bf=net_brake(name,grade);bp=net_brake(pred,grade)
            gap0=CONFIG['standstill_gap_m']+h*v
            raw={
              'spacing':gap0-CONFIG['physical_gap_min_m']-CONFIG['model_gap_reserve_m'],
              'stopping':gap0-CONFIG['stopping_reserve_m']-v*(age+p.tau)-v*v/(2*bf)+v*v/(2*bp)-CONFIG['model_stop_reserve_m'] if bf>0 and bp>0 else -math.inf,
              'speed_envelope':speed_limit(curvature)-v-CONFIG['model_speed_reserve_mps'],
              'brake_input':bf/p.b-CONFIG['model_brake_reserve_mps2'],
              'terminal':CONFIG['terminal_error_halfwidth_m']-CONFIG['model_gap_reserve_m'],
            }
            row={}
            for field,output in [('spacing','gap'),('speed_envelope','v'),('brake_input','input'),('terminal','e')]:
                row[field]=tuple(m[output+'_'+t][i][k,:] for t in ('u','eps','w'))
            if bf>0 and bp>0:
                row['stopping']=tuple(m['gap_'+t][i][k,:]
                    -(age+p.tau+v/bf)*m['v_'+t][i][k,:]
                    +(v/bp)*m['pred_v_'+t][i][k,:] for t in ('u','eps','w'))
            else:row['stopping']=(np.zeros(H),np.zeros(3*H*(N-1)),np.zeros(H*N))
            for field in FIELDS:
                if field=='terminal' and k!=H-1:continue
                lu,ee,ww=row[field]
                M=float(raw[field]-LEADER_L2_BUDGET*np.linalg.norm(lu)-W_L2_BUDGET*np.linalg.norm(ww))
                yield dict(constraint=field,follower=i,step=k,margin=M,
                           nominal=float(raw[field]),leader_reserve=float(LEADER_L2_BUDGET*np.linalg.norm(lu)),
                           disturbance_reserve=float(W_L2_BUDGET*np.linalg.norm(ww))),ee

def compute(condition,keep_vectors=False):
    kind=condition['kind'];load=condition['load'];grade=condition['grade'];curve=condition['curvature']
    h=float(condition['headway_s']);age=float(condition['age_s'])
    metadata=[];vectors=[];mins=np.full(3,np.inf);families=['','','']
    all_positive=True;ball=np.inf;critical_field='';min_margin=np.inf
    for meta,f in row_maps(kind,load,grade,curve,h,age):
        M=meta['margin'];min_margin=min(min_margin,M)
        if M<=0:all_positive=False
        if M>0:
            norm=float(np.linalg.norm(f));crit=M/(norm*EPS_STACK_FACTOR) if norm else np.inf
            if crit<ball:ball=crit;critical_field=meta['constraint']
            sensitivity=np.array([np.sum(np.abs(f[c::3])) for c in range(3)])
            for c in range(3):
                q=M/sensitivity[c] if sensitivity[c]>0 else np.inf
                if q<mins[c]:mins[c]=q;families[c]=meta['constraint']
        if keep_vectors:
            metadata.append(meta);vectors.append(f.copy())
    result=dict(**condition,feasible=int(all_positive),min_margin=min_margin,
        epsilon_crit_recomputed=ball if all_positive else float('nan'),
        scalar_active=critical_field if all_positive else '',
        q_gap_crit=mins[0] if all_positive else float('nan'),
        q_vrel_crit=mins[1] if all_positive else float('nan'),
        q_apred_crit=mins[2] if all_positive else float('nan'),
        gap_axis_active=families[0] if all_positive else '',
        vrel_axis_active=families[1] if all_positive else '',
        apred_axis_active=families[2] if all_positive else '')
    if keep_vectors:
        return result,np.array(vectors,dtype=np.float64),metadata
    return result

def main():
    source=OUT/'common_input_certificate_table.csv'
    with source.open(newline='',encoding='utf-8') as f:original=list(csv.DictReader(f))
    rows=[]
    for j,r in enumerate(original):
        condition={k:r[k] for k in ('kind','load','grade','curvature','headway_s','age_s')}
        out=compute(condition);rows.append(out)
        if j%100==0:print('axis',j,flush=True)
    with (OUT/'directional_axes_517.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for condition in SELECTED:
        summary,F,meta=compute(condition,True)
        stem=condition['name']
        np.savez_compressed(OUT/f'directional_rows_{stem}.npz',F=F,M=np.array([r['margin'] for r in meta]),
                            constraint=np.array([r['constraint'] for r in meta]),
                            follower=np.array([r['follower'] for r in meta]),step=np.array([r['step'] for r in meta]))
        print('selected',stem,summary['epsilon_crit_recomputed'],summary['q_gap_crit'],summary['q_vrel_crit'],summary['q_apred_crit'],flush=True)
    (OUT/'directional_protocol.json').write_text(json.dumps(dict(selected=SELECTED,H=H,dt=CONFIG['dt'],
        constraints=FIELDS,source='frozen v3 common-input lifted model',
        reference_units=['1 m','1 m/s','1 m/s^2'],
        axis_interpretation='one channel bounded at every sample for all three followers; other channels zero',
        no_test_tuning=True),indent=2))

if __name__=='__main__':main()
