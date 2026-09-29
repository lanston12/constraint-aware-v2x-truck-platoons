"""Rebuild v7 figures and their compact, inspectable plot sources."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from v7_specs import OUT, BUDGETS, SCENARIOS

ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
NAMES=[s['name'] for s in SCENARIOS]
LABEL={'ap':'AP','aoi':'AoI','state':'State','cr_delta_c':'CR-DeltaC','cr_voi':'CR-VoI'}
COL={'ap':'#ad4d32','aoi':'#b88b2f','state':'#365f98','cr_delta_c':'#268c7a','cr_voi':'#552f80'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42})

def save(fig,name):
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(FIG/(name+'.png'),dpi=210,bbox_inches='tight')
    plt.close(fig)

def fig7(p):
    src=p[['scenario','seed','budget_kb','reference_row_entropy',
           'reference_switching_count','reference_modal_active_family',
           'delta_mean_c_cr_voi_minus_state','delta_mean_c_cr_delta_c_minus_state',
           'delta_mean_c_cr_delta_c_minus_cr_voi']]
    src.to_csv(OUT/'fig7_diversity_vs_advantage_raw.csv',index=False)
    fig,axs=plt.subplots(2,2,figsize=(10.5,6.5),constrained_layout=True)
    axs=axs.flat
    for name,g in src.groupby('scenario'):
        i=NAMES.index(name)
        axs[0].scatter(g.reference_row_entropy,g.delta_mean_c_cr_voi_minus_state,
                       s=12,alpha=.55,label=name,marker=['o','s','^','D','v','P'][i])
        axs[1].scatter(g.reference_switching_count,g.delta_mean_c_cr_delta_c_minus_state,
                       s=12,alpha=.45,marker=['o','s','^','D','v','P'][i])
    for ax in axs[:2]:ax.axhline(0,color='black',lw=.7)
    axs[0].set(xlabel='Policy-neutral row entropy',ylabel=r'$C_{CR-VoI}-C_{State}$',
               title='A  Directional score vs state uncertainty')
    axs[1].set(xlabel='Policy-neutral row switches / 80 slots',
               ylabel=r'$C_{CR-\Delta C}-C_{State}$',title='B  Switching vs state gap')
    axs[0].legend(fontsize=6,loc='best',ncol=2)
    groups=src.groupby('reference_modal_active_family')
    fam=list(groups.groups)
    data=[groups.get_group(f).delta_mean_c_cr_voi_minus_state.values for f in fam]
    axs[2].boxplot(data,labels=fam,showfliers=False)
    axs[2].axhline(0,color='black',lw=.7)
    axs[2].tick_params(axis='x',rotation=35)
    axs[2].set(ylabel=r'$C_{CR-VoI}-C_{State}$',title='C  Reference modal bottleneck')
    bins=[-.001,.001,.55,.7]
    names=['H=0','0<H≤.55','H>.55']
    cat=pd.cut(src.reference_row_entropy,bins=bins,labels=names)
    count=[]
    for name in names:
        g=src[cat==name]
        count.append(dict(diversity_bin=name,n=len(g),
                          voi_win_rate=float((g.delta_mean_c_cr_voi_minus_state<0).mean()) if len(g) else np.nan,
                          delta_win_rate=float((g.delta_mean_c_cr_delta_c_minus_state<0).mean()) if len(g) else np.nan))
    pd.DataFrame(count).to_csv(OUT/'fig7_diversity_win_rates.csv',index=False)
    xx=np.arange(len(names));width=.38
    axs[3].bar(xx-width/2,[r['voi_win_rate'] for r in count],width,label='CR-VoI',color=COL['cr_voi'])
    axs[3].bar(xx+width/2,[r['delta_win_rate'] for r in count],width,label='CR-DeltaC',color=COL['cr_delta_c'])
    axs[3].set(xticks=xx,xticklabels=[f'{r["diversity_bin"]}\nN={r["n"]}' for r in count],
               ylim=(0,1),ylabel='Paired win fraction vs State',title='D  Win rate by entropy bin')
    axs[3].legend(fontsize=7)
    save(fig,'fig7_diversity_vs_advantage')

def fig8(p):
    p=p.copy()
    order=['H=0','0<H≤.55','H>.55']
    p['diversity_bin']=pd.cut(p.reference_row_entropy,
        bins=[-.001,.001,.55,.7],labels=order)
    g=p.groupby(['diversity_bin','budget_kb'],as_index=False,observed=True).agg(
        voi_state=('delta_mean_c_cr_voi_minus_state','mean'),
        voi_ap=('delta_mean_c_cr_voi_minus_ap','mean'),
        delta_state=('delta_mean_c_cr_delta_c_minus_state','mean'),
        row_entropy=('reference_row_entropy','mean'),n=('seed','size'))
    g.to_csv(OUT/'fig8_constraint_regime_map_raw.csv',index=False)
    means=p.groupby(['diversity_bin','budget_kb'],observed=True)[[f'mean_c_{m}' for m in LABEL]].mean()
    best=means.idxmin(axis=1).str.replace('mean_c_','',regex=False).rename('best_method').reset_index()
    best.to_csv(OUT/'fig8_best_method_raw.csv',index=False)
    fig,axs=plt.subplots(3,2,figsize=(10.4,8.7),constrained_layout=True)
    for ax,(field,title) in zip(axs.flat,[('voi_state','CR-VoI minus State'),
                                         ('voi_ap','CR-VoI minus AP'),
                                         ('delta_state','CR-DeltaC minus State'),
                                         ('row_entropy','Reference row entropy')]):
        mat=g.pivot(index='diversity_bin',columns='budget_kb',values=field).reindex(
            index=order,columns=BUDGETS).values
        if field=='row_entropy':im=ax.imshow(mat,cmap='Blues',aspect='auto',vmin=0)
        else:
            lim=max(abs(np.nanmin(mat)),abs(np.nanmax(mat)),.01)
            im=ax.imshow(mat,cmap='RdBu_r',aspect='auto',norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim))
        for i in range(len(order)):
            for j in range(len(BUDGETS)):
                value=mat[i,j]
                label=('—' if not np.isfinite(value) else
                       (f'{value:+.2f}' if field!='row_entropy' else f'{value:.2f}'))
                ax.text(j,i,label,
                        ha='center',va='center',fontsize=6)
        ax.set(xticks=range(len(BUDGETS)),xticklabels=[f'{x:g}' for x in BUDGETS],
               yticks=range(len(order)),yticklabels=order,xlabel='Offered cap (kbit/slot)',
               title=title)
        fig.colorbar(im,ax=ax,shrink=.74)
    code={m:i for i,m in enumerate(LABEL)}
    mat=best.assign(code=best.best_method.map(code)).pivot(index='diversity_bin',columns='budget_kb',values='code').reindex(index=order,columns=BUDGETS).values
    ax=axs.flat[4];ax.imshow(mat,cmap=matplotlib.colors.ListedColormap([COL[m] for m in LABEL]),vmin=-.5,vmax=4.5,aspect='auto')
    for i in range(len(order)):
        for j in range(len(BUDGETS)):
            label='—' if not np.isfinite(mat[i,j]) else LABEL[list(LABEL)[int(mat[i,j])]]
            ax.text(j,i,label,ha='center',va='center',fontsize=5,color='white')
    ax.set(xticks=range(len(BUDGETS)),xticklabels=[f'{x:g}' for x in BUDGETS],yticks=range(len(order)),yticklabels=order,xlabel='Offered cap (kbit/slot)',title='Best mean C of five methods')
    axs.flat[5].axis('off')
    axs.flat[5].text(.02,.9,'Negative contrast favors CR.\nAll cells retain 12 paired seeds.\nRow diversity uses a policy-neutral reference.',va='top')
    save(fig,'fig8_constraint_regime_map')

def fig9(ep,p,rank):
    a=ep[ep.method.isin(['cr_delta_c','cr_voi'])].copy()
    a.to_csv(OUT/'fig9_cr_denominator_episodes_raw.csv',index=False)
    rank.to_csv(OUT/'fig9_ranking_disagreement_raw.csv',index=False)
    fig,axs=plt.subplots(3,2,figsize=(10,9),constrained_layout=True)
    for method,g in a.groupby('method'):
        c=COL[method]
        axs[0,0].scatter(g.received_kbit_per_s,g.mean_c,s=9,alpha=.16,color=c)
        mean=g.groupby('budget_kb')[['received_kbit_per_s','mean_c']].mean()
        axs[0,0].plot(mean.received_kbit_per_s,mean.mean_c,'o-',lw=1.3,color=c,label=LABEL[method])
        axs[0,1].scatter(g.received_kbit_per_s,g.minimum_model_margin,s=9,alpha=.16,color=c)
        axs[0,1].plot(mean.received_kbit_per_s,
                      g.groupby('budget_kb').minimum_model_margin.mean(),'o-',lw=1.3,color=c)
    axs[0,0].set(xlabel='Actual received (kbit/s)',ylabel='Mean model C',title='A  Received-bit control Pareto')
    axs[0,0].legend()
    axs[0,1].set(xlabel='Actual received (kbit/s)',ylabel='Minimum model margin',title='B  Margin and received bits')
    dg=p.groupby('budget_kb').delta_mean_c_cr_delta_c_minus_cr_voi.agg(['mean','std','count'])
    axs[1,0].errorbar(dg.index,dg['mean'],yerr=1.96*dg['std']/np.sqrt(dg['count']),fmt='o-',color=COL['cr_delta_c'])
    axs[1,0].axhline(0,color='black',lw=.7)
    axs[1,0].set(xlabel='Offered cap (kbit/slot)',ylabel=r'$C_{CR-\Delta C}-C_{CR-VoI}$',
                  title='C  Denominator ablation, ±1.96 descriptive SE')
    dr=rank.groupby('budget_kb').cr_rank_disagreement.mean()
    axs[1,1].plot(dr.index,dr.values,'o-',color='#6b8292',label='Closed-loop first packet')
    common=pd.read_csv(OUT/'denominator_common_state_ranking.csv')
    for field,label,color in [('raw_vs_bit_only','Raw vs bit-only','#bf7e30'),
                              ('bit_only_vs_full','Bit-only vs full','#338d87'),
                              ('raw_vs_full','Raw vs full','#674298')]:
        line=common.groupby('budget_kb')[field].mean()
        axs[1,1].plot(line.index,line.values,'o-',lw=1,color=color,label=label)
    axs[1,1].set(xlabel='Offered cap (kbit/slot)',ylabel='First-packet disagreement fraction',
                  ylim=(0,1),title='D  Common-state denominator components')
    axs[1,1].legend(fontsize=6,loc='lower right')
    xx=np.arange(6);sizes=[3,5,8,10,16,24]
    for j,method in enumerate(['cr_delta_c','cr_voi']):
        g=a[a.method==method]
        counts=np.array([g[f'packets_{b}kb'].sum() for b in sizes],float)
        axs[2,0].bar(xx+(.38*j),counts/counts.sum(),.38,color=COL[method],label=LABEL[method])
    axs[2,0].set(xticks=xx+.19,xticklabels=[str(b) for b in sizes],xlabel='Packet size (kbit)',
                  ylabel='Fraction of offered packets',title='E  Packet-size distribution')
    axs[2,0].legend(fontsize=7)
    for method,g in a.groupby('method'):
        avg=g.groupby('budget_kb').received_kbit_per_s.mean()
        axs[2,1].plot(avg.index,avg.values,'o-',color=COL[method],label=LABEL[method])
    axs[2,1].set(xlabel='Offered cap (kbit/slot)',ylabel='Actual received (kbit/s)',
                  title='F  Policy-dependent delivered rate')
    save(fig,'fig9_cr_denominator_ablation')

def fig10(ref,slots):
    chosen=ref.sort_values(['switching_count','row_entropy','scenario','seed'],
                           ascending=[False,False,True,True]).iloc[0]
    name=chosen.scenario;seed=int(chosen.seed);budget=10.
    sel=slots[(slots.scenario==name)&(slots.seed==seed)&
              (slots.budget_kb==budget)&slots.method.isin(['state','cr_delta_c','cr_voi'])]
    if sel.empty:return None
    sel.to_csv(OUT/'fig10_switching_trace_raw.csv',index=False)
    reference_slots=pd.read_csv(OUT/'reference_slot_metrics.csv')
    reference_slots=reference_slots[(reference_slots.scenario==name)&(reference_slots.seed==seed)]
    reference_slots.to_csv(OUT/'fig10_switching_reference_raw.csv',index=False)
    pd.DataFrame([dict(scenario=name,seed=seed,budget_kb=budget,
                       selection='max reference switches, then entropy, then scenario/seed',
                       switches=int(chosen.switching_count),entropy=chosen.row_entropy)]).to_csv(
                           OUT/'fig10_switching_selection.csv',index=False)
    fig,axs=plt.subplots(9,1,figsize=(10,12),sharex=True,constrained_layout=True)
    axs[0].step(reference_slots.time_s,reference_slots.active_family,where='post',
                color='black',ls='--',lw=1.5,label='Policy-neutral reference',zorder=5)
    for method,g in sel.groupby('method'):
        g=g.sort_values('step');t=g.time_s;col=COL[method]
        axs[0].step(t,g.active_family,where='post',color=col,alpha=.7,label=LABEL[method])
        axs[1].plot(t,g.state_norm_after,color=col,lw=1,label=LABEL[method])
        axs[2].plot(t,g.delta_c,color=col,lw=1)
        axs[3].step(t,g.first_action_id,where='mid',color=col,lw=.75)
        axs[4].plot(t,g.c_after,color=col,lw=1)
        axs[5].plot(t,g.model_margin,color=col,lw=1)
        axs[6].step(t,g.received_bits_kb,where='mid',color=col,lw=.9)
        axs[7].plot(t,g.aoi_s,color=col,lw=1)
        axs[8].plot(t,g.spacing_rmse_step_m,color=col,lw=1)
    axs[0].set_ylabel('Active family');axs[0].legend(ncol=3,fontsize=7)
    for ax,label in zip(axs[1:],['State norm','Slot ΔC','First packet ID','Model C','1 − C','Received kbit','AoI (s)','Spacing RMSE (m)']):ax.set_ylabel(label)
    axs[-1].set_xlabel('Time (s)')
    fig.suptitle(f'Automatically selected switching trace: {name}, seed {seed}, cap 10')
    save(fig,'fig10_switching_mechanism')
    return name,seed,int(chosen.switching_count)

if __name__=='__main__':
    ep=pd.read_csv(OUT/'episode_metrics.csv')
    p=pd.read_csv(OUT/'paired_contrasts.csv')
    rank=pd.read_csv(OUT/'ranking_disagreement.csv')
    ref=pd.read_csv(OUT/'reference_diversity.csv')
    slots=pd.read_csv(OUT/'slot_metrics.csv')
    fig7(p);fig8(p);fig9(ep,p,rank)
    print('fig10:',fig10(ref,slots))
