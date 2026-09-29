"""Unified paired budget x stress sweep using the frozen v5 algorithm and plant."""
import csv
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from scipy.stats import t
from v5_horizon import SCENARIOS, condition_map, write_csv
from v5_sim import ROOT, NetworkPlatoonEnv, summarize, row_sensitivities

OUT = ROOT / 'results/v6'
OUT.mkdir(parents=True, exist_ok=True)
BUDGETS = (5., 8., 10., 12., 16., 24.)
SEEDS = tuple(range(201, 213))
METHODS = ('ap', 'aoi', 'state', 'cr_voi_1')
FAMILIES = ('brake_input', 'stopping', 'spacing', 'speed_envelope', 'terminal')

def run_stress(scenario):
    rows = []
    _, _, step, labels = condition_map(next(s for s in SCENARIOS if s['name'] == scenario))
    labels = labels[step < 1]
    for budget in BUDGETS:
        for seed in SEEDS:
            for method in METHODS:
                env = NetworkPlatoonEnv(scenario, budget, seed)
                counts = Counter()
                while env.k < 80:
                    decision = env.rank(method)
                    env.step(decision, method)
                    if method == 'cr_voi_1':
                        active = labels[int(np.argmax(env.G2[0] @ env.sigma.reshape(9)**2))]
                        counts[str(active)] += 1
                summary = summarize(env.rows)
                summary.update({f'active_{x}_steps': counts[x] for x in FAMILIES})
                rows.append(summary)
        print('completed', scenario, budget, flush=True)
    return rows

def interval(d):
    d = np.asarray(d, float)
    half = t.ppf(.975, len(d)-1) * d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean()), float(d.mean()-half), float(d.mean()+half)

def main():
    scenarios = tuple(s['name'] for s in SCENARIOS)
    # Finish cache creation before parallel reads.
    for s in SCENARIOS:
        condition_map(s)
        row_sensitivities(s['name'])
    protocol = dict(
        evidence='model-based synthetic network/perception and frozen v4 rows',
        base_model='v5 simulator, no scheduler or controller algorithm change',
        scenarios=[dict(s) for s in SCENARIOS], budgets_kbit_per_slot=BUDGETS,
        paired_seeds=SEEDS, methods=METHODS, dt_s=.1, steps=80,
        comparison_unit='one complete paired seed episode',
        shared_per_cell='candidate actions, network/occlusion/error draws, plant, controller, cap',
        paired_contrast='C_CR-C_comparator; negative favors CR',
        active_constraint='modal one-step modeled row under CR, pooled over all 80 slots and paired seeds',
        AP='smooth model proxy, not detector AP@IoU',
        received_bits='policy-dependent and preserved')
    (OUT/'regime_protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    allrows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(run_stress, s): s for s in scenarios}
        for future in as_completed(futures):
            allrows.extend(future.result())
            print('collected', futures[future], len(allrows), flush=True)
    allrows.sort(key=lambda r:(scenarios.index(r['scenario']),float(r['budget_kb']),
                               int(r['seed']),METHODS.index(r['method'])))
    write_csv(OUT/'regime_episode_metrics.csv', allrows)
    contrasts = []
    active = []
    pairs = []
    for scenario in scenarios:
        for budget in BUDGETS:
            subset = [r for r in allrows if r['scenario']==scenario and
                      float(r['budget_kb'])==budget]
            by = {(r['method'],int(r['seed'])):r for r in subset}
            mean_c = {m:float(np.mean([float(by[m,s]['mean_c']) for s in SEEDS]))
                      for m in METHODS}
            best = min(METHODS,key=lambda m:(mean_c[m],METHODS.index(m)))
            counts = {f:sum(int(by['cr_voi_1',s][f'active_{f}_steps'])
                            for s in SEEDS) for f in FAMILIES}
            modal = max(FAMILIES,key=lambda f:counts[f])
            active.append(dict(scenario=scenario,budget_kb=budget,
                               best_method=best,active_constraint=modal,
                               **{f'{f}_fraction':counts[f]/(80*len(SEEDS))
                                  for f in FAMILIES}))
            for comparator in ('ap','aoi','state'):
                diffs = [float(by['cr_voi_1',s]['mean_c'])-
                         float(by[comparator,s]['mean_c']) for s in SEEDS]
                mean,lo,hi=interval(diffs)
                contrasts.append(dict(scenario=scenario,budget_kb=budget,
                    comparator=comparator,n=len(SEEDS),
                    mean_delta_c=mean,ci95_low=lo,ci95_high=hi,
                    cr_advantage=int(hi<0),comparator_advantage=int(lo>0),
                    best_method=best,active_constraint=modal))
            for seed in SEEDS:
                ap,cr=by['ap',seed],by['cr_voi_1',seed]
                x=float(ap['mean_ap_proxy'])-float(cr['mean_ap_proxy'])
                y=float(cr['minimum_active_constraint_margin'])-float(
                    ap['minimum_active_constraint_margin'])
                pairs.append(dict(scenario=scenario,budget_kb=budget,seed=seed,
                    ap_proxy_advantage_ap=x,control_margin_advantage_cr=y,
                    disagreement=int(x>0 and y>0),
                    ap_proxy_ap=ap['mean_ap_proxy'],ap_proxy_cr=cr['mean_ap_proxy'],
                    minimum_margin_ap=ap['minimum_active_constraint_margin'],
                    minimum_margin_cr=cr['minimum_active_constraint_margin'],
                    mean_c_ap=ap['mean_c'],mean_c_cr=cr['mean_c'],
                    received_kbit_per_s_ap=ap['received_kbit_per_s'],
                    received_kbit_per_s_cr=cr['received_kbit_per_s']))
    write_csv(OUT/'regime_contrasts.csv',contrasts)
    write_csv(OUT/'regime_active_best.csv',active)
    write_csv(OUT/'ap_control_disagreement.csv',pairs)
    print('episodes',len(allrows),'paired AP/CR',len(pairs),
          'disagreement',sum(x['disagreement'] for x in pairs),flush=True)

if __name__=='__main__':
    main()
