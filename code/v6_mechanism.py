"""Fixed inherited seed 201; select divergent moments by a declared maximum rule."""
import json
from pathlib import Path
import numpy as np
from v5_horizon import condition_map, SCENARIOS, write_csv
from v5_sim import ROOT, ACTIONS, NetworkPlatoonEnv

OUT=ROOT/'results/v6'
OUT.mkdir(exist_ok=True)
SCENARIO='combined'
BUDGET=10.
SEED=201  # inherited v5 Fig. 6 trace, fixed before v6 event selection
METHODS=('ap','aoi','cr_voi_1')
scenario=next(s for s in SCENARIOS if s['name']==SCENARIO)
_,_,step,alllabels=condition_map(scenario)
labels=alllabels[step<1]

def describe(a):
    if a.id==0:return 'skip'
    return (f'{a.source}>{a.receiver} '+
            ('S' if a.packet=='state' else 'F')+'/'+
            ('N' if a.compression=='none' else
             'M' if a.compression=='medium' else 'H')+
            f' ch{a.channel}')

def main():
    envs={m:NetworkPlatoonEnv(SCENARIO,BUDGET,SEED) for m in METHODS}
    traces=[];candidates=[];events=[]
    for k in range(80):
        decisions={m:envs[m].rank(m) for m in METHODS}
        ref=envs['cr_voi_1']
        sigma=ref.sigma.copy()
        before_c=ref.cbar(sigma)
        before_ap=ref.ap_proxy(sigma)
        metrics=[]
        for m in METHODS:
            aid=int(decisions[m][0])
            a=ACTIONS[aid]
            after,p,lat=ref.action_model(a,sigma)
            gain=ref.ap_proxy(after)-before_ap
            aoi_red=0. if aid==0 else p*max(0.,ref.aoi[a.receiver-1]-lat)
            state_red=float(np.linalg.norm(sigma)-np.linalg.norm(after))
            delta_c=before_c-ref.cbar(after)
            active=labels[int(np.argmax(ref.G2[0] @ after.reshape(9)**2))]
            row=dict(time_s=round((k+1)*.1,3),step=k+1,seed=SEED,
                     scenario=SCENARIO,budget_kb=BUDGET,method=m,
                     candidate_action_id=aid,candidate_packet=describe(a),
                     source=a.source,receiver=a.receiver,packet=a.packet,
                     compression=a.compression,channel=a.channel,
                     expected_ap_proxy_gain=gain,
                     expected_aoi_reduction_s=aoi_red,
                     expected_state_norm_reduction=state_red,
                     expected_delta_c=delta_c,bits_kb=a.bits_kb,
                     latency_ms=1000*lat,delivery_probability=p,
                     active_row_after=str(active),
                     reference_state='CR pre-decision state; common for all 3 candidates')
            candidates.append(row);metrics.append(row)
        unique=len(set(r['candidate_action_id'] for r in metrics))
        score=(max(r['expected_delta_c'] for r in metrics)-
               min(r['expected_delta_c'] for r in metrics)) if unique>=2 else 0.
        events.append(dict(step=k+1,time_s=round((k+1)*.1,3),
                           distinct_first_actions=unique,
                           delta_c_range=score))
        for m in METHODS:
            row,_=envs[m].step(decisions[m],m)
            env=envs[m]
            active=labels[int(np.argmax(env.G2[0] @ env.sigma.reshape(9)**2))]
            traces.append({**row,'active_constraint':str(active),
                           'first_packet':describe(ACTIONS[int(decisions[m][0])]),
                           'first_action_id':int(decisions[m][0])})
    eligible=sorted((r for r in events if r['distinct_first_actions']>=2),
                    key=lambda r:(-r['delta_c_range'],r['step']))
    chosen=[]
    for r in eligible:
        if all(abs(r['step']-q['step'])>=5 for q in chosen):
            chosen.append(r)
        if len(chosen)==2:break
    chosen.sort(key=lambda r:r['step'])
    if len(chosen)<2:raise ValueError('Fewer than two divergent events')
    write_csv(OUT/'mechanism_trace.csv',traces)
    write_csv(OUT/'mechanism_candidates.csv',candidates)
    write_csv(OUT/'mechanism_all_event_scores.csv',events)
    write_csv(OUT/'mechanism_selected_events.csv',chosen)
    protocol=dict(evidence='synthetic combined closed-loop model',
                  inherited_fixed_seed=SEED,budget_kbit_per_slot=BUDGET,
                  methods=METHODS,
                  event_rule='top two distinct-first-action time slots by range of expected single-packet ΔC, at least five slots apart; tie earliest',
                  candidate_reference='CR pre-decision model state for all method-selected first packets',
                  action_metrics='expected single-packet effects, not realized whole-slot effects',
                  abbreviations='S state, F feature; N none, M medium, H high compression')
    (OUT/'mechanism_protocol.json').write_text(
        json.dumps(protocol,indent=2),encoding='utf-8')
    print('selected',chosen)
    for r in candidates:
        if r['step'] in [q['step'] for q in chosen]:
            print(r['time_s'],r['method'],r['candidate_packet'],
                  round(r['expected_delta_c'],4),r['active_row_after'])

if __name__=='__main__':main()
