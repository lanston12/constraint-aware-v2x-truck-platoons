"""Reproduce figures, headline aggregates, and a fixed paired episode."""
from pathlib import Path
import os, sys, json, hashlib, runpy
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'code'))
import numpy as np
import pandas as pd
from scipy.stats import t, spearmanr

def verify_inputs():
    entries = json.loads((ROOT/'frozen_inputs_manifest.json').read_text(encoding='utf-8'))
    for entry in entries:
        actual = hashlib.sha256((ROOT/entry['path']).read_bytes()).hexdigest()
        if actual != entry['sha256']:
            raise ValueError('Input checksum mismatch: '+entry['path'])
    return len(entries)

def read(path):
    return pd.read_csv(ROOT/'results'/path)

def statistics():
    ep = read('v6/regime_episode_metrics.csv')
    keys = ['scenario','budget_kb','seed']
    assert len(ep) == 1728 and not ep.duplicated(keys+['method']).any()
    wide = ep.pivot(index=keys, columns='method', values='mean_c')
    assert wide.shape == (432,4) and not wide.isna().any().any()
    contrasts = []
    for (scenario,budget), group in wide.groupby(level=[0,1]):
        assert len(group) == 12
        for method in ('ap','aoi','state'):
            diff = group.cr_voi_1-group[method]
            half = t.ppf(.975,11)*diff.std(ddof=1)/np.sqrt(12)
            contrasts.append(dict(scenario=scenario,budget_kb=budget,comparator=method,
                mean_delta_c=diff.mean(),ci95_low=diff.mean()-half,ci95_high=diff.mean()+half))
    contrasts = pd.DataFrame(contrasts)
    index = ['scenario','budget_kb','comparator']
    fields = ['mean_delta_c','ci95_low','ci95_high']
    np.testing.assert_allclose(contrasts.set_index(index)[fields].sort_index(),
        read('v6/regime_contrasts.csv').set_index(index)[fields].sort_index(),rtol=1e-10,atol=1e-12)
    contrasts.to_csv(ROOT/'audit/recomputed_regime_contrasts.csv',index=False)
    ap = ep[ep.method=='ap'].set_index(keys)
    cr = ep[ep.method=='cr_voi_1'].set_index(keys)
    disagreement = pd.DataFrame({
        'ap_proxy_advantage_ap':ap.mean_ap_proxy-cr.mean_ap_proxy,
        'control_margin_advantage_cr':cr.minimum_active_constraint_margin-ap.minimum_active_constraint_margin})
    disagreement['disagreement'] = (disagreement.ap_proxy_advantage_ap>0)&(disagreement.control_margin_advantage_cr>0)
    disagreement['score'] = disagreement.ap_proxy_advantage_ap.clip(lower=0)*disagreement.control_margin_advantage_cr.clip(lower=0)
    disagreement.to_csv(ROOT/'audit/recomputed_disagreement.csv')
    counts = disagreement.groupby(level=[0,1]).disagreement.agg(['sum','count','mean'])
    counts.to_csv(ROOT/'audit/disagreement_by_stress_and_budget.csv')
    selected = disagreement.reset_index().sort_values(['score','scenario','budget_kb','seed'],ascending=[False,True,True,True]).iloc[0]
    points = read('v6/received_pareto_points.csv')
    fields = ['received_kbit_per_s','mean_c','minimum_active_constraint_margin','mean_ap_proxy','spacing_rmse_m']
    # Check every saved Pareto point against its episode-level source.
    for source, p in points.groupby('source'):
        base = ep if source=='v6_full_sweep' else read('v5/deterministic_episode_metrics.csv')
        p = p.set_index(keys+['method']).sort_index()
        b = base.set_index(keys+['method']).loc[p.index]
        np.testing.assert_allclose(p[fields],b[fields],rtol=1e-10,atol=1e-12)
    means = points.groupby(['scenario','budget_kb','method'])[fields].mean()
    saved = read('v6/received_pareto_cell_summary.csv').set_index(['scenario','budget_kb','method']).sort_index()
    np.testing.assert_allclose(means.sort_index(),saved[[f+'_mean' for f in fields]],rtol=1e-10,atol=1e-12)
    means.to_csv(ROOT/'audit/recomputed_received_pareto_means.csv')
    fresh = read('calibration/fresh_certificate_comparison.csv')
    fresh = fresh[fresh.operating_condition=='light_brake']
    accepted = fresh[fresh.accepted==1]
    covered = int((accepted.C_true<=accepted.U4_projected).sum())
    rho = float(spearmanr(fresh.state_norm,fresh.C_true)[0])
    axes = read('certificates/common_input_certificate_table.csv')
    h80 = read('v5/horizon_metrics.csv')
    h80 = h80[h80.H==80][['scenario','qualified_fraction','median_Cbar','coverage']]
    h80.to_csv(ROOT/'audit/recomputed_h80_summary.csv',index=False)
    report = dict(episodes=len(ep),paired_episodes=len(wide),
        cr_lower_mean_c_cells={m:int(((contrasts.comparator==m)&(contrasts.mean_delta_c<0)).sum()) for m in ('ap','aoi','state')},
        disagreement_count=int(disagreement.disagreement.sum()),disagreement_fraction=float(disagreement.disagreement.mean()),
        maximum_disagreement=dict(scenario=selected.scenario,budget_kb=float(selected.budget_kb),seed=int(selected.seed),score=float(selected.score)),
        empirical=dict(matched=len(fresh),accepted=len(accepted),direct_bound_covered=covered,spearman_rho=rho),
        control_geometry=dict(cases=len(axes),positive_radius_cases=int((axes.epsilon_crit>0).sum())),
        mechanism_times_s=read('v6/mechanism_selected_events.csv').time_s.tolist(),
        supplementary_episodes=len(read('v7/episode_metrics.csv')))
    assert covered==212 and len(accepted)==223
    assert report['disagreement_count']==156
    (ROOT/'audit/headline_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

def replay():
    from v5_sim import NetworkPlatoonEnv, summarize
    frozen = read('v6/regime_episode_metrics.csv')
    rows = []
    traces = []
    for method in ('ap','aoi','state','cr_voi_1'):
        env = NetworkPlatoonEnv('combined',10.,201)
        while env.k<80:
            env.step(env.rank(method),method)
        actual = summarize(env.rows)
        ref = frozen[(frozen.scenario=='combined')&(frozen.budget_kb==10)&(frozen.seed==201)&(frozen.method==method)].iloc[0]
        errors = []
        for key,value in actual.items():
            if isinstance(value,(float,int,np.number)):
                np.testing.assert_allclose(value,ref[key],rtol=1e-9,atol=1e-10,err_msg=method+': '+key)
                errors.append(abs(float(value)-float(ref[key])))
        rows.append(dict(method=method,maximum_metric_difference=max(errors),status='PASS'))
        traces.extend(env.rows)
    pd.DataFrame(traces).to_csv(ROOT/'audit/replayed_combined_seed201_trace.csv',index=False)
    (ROOT/'audit/episode_replay.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    return rows

def main():
    (ROOT/'audit').mkdir(exist_ok=True)
    (ROOT/'figures').mkdir(exist_ok=True)
    count = verify_inputs()
    print('Verified',count,'frozen source/data files.',flush=True)
    # The existing wrapper prevents any overwriting of frozen result files.
    from submission_plots import plot, supplement_plots
    results = statistics()
    print(json.dumps(results,indent=2),flush=True)
    runpy.run_path(str(ROOT/'code/final_lift_check.py'),run_name='__main__')
    print('Fixed Fig.6 episode replay:',json.dumps(replay()),flush=True)
    for name in ['v5_plot_relevance.py','v5_plot_horizon.py','v6_plot_pareto.py',
                 'v6_plot_regime.py','v6_plot_mechanism.py','v6_plot_disagreement.py','v5_plot_gsppo.py']:
        plot(name)
    supplement_plots()
    expected = ['fig2_control_relevance','fig3_information_horizon','fig4_received_bit_pareto',
                'fig5_regime_map','fig6_combined_mechanism','fig_s1_ap_control_disagreement',
                'fig5_gsppo_comparison','fig7_diversity_vs_advantage','fig8_constraint_regime_map',
                'fig9_cr_denominator_ablation','fig10_switching_mechanism']
    for stem in expected:
        for ext in ('pdf','png'):
            assert (ROOT/'figures'/f'{stem}.{ext}').stat().st_size>1000
    assert verify_inputs()==count
    import scipy, matplotlib
    (ROOT/'audit/reproduction_status.json').write_text(json.dumps(dict(status='PASS',
        regenerated_result_figures=11,preserved_schematic='figures/fig1.png',
        frozen_input_files_verified=count,paired_replayed_episodes=4,
        versions=dict(python=sys.version.split()[0],numpy=np.__version__,pandas=pd.__version__,
                      scipy=scipy.__version__,matplotlib=matplotlib.__version__)),indent=2),encoding='utf-8')
    print('PASS: 11 result figures (PDF/PNG), statistics, lift consistency, four paired episode replays.',flush=True)

if __name__=='__main__':
    main()
