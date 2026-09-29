"""Publication-size CPH figure from saved trajectory-level simulation results."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'results/v5'
FIG = ROOT/'figures'
def read(name):
    with (DATA/name).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

metrics = read('horizon_metrics.csv')
detail = read('horizon_trajectories.csv')
active = read('horizon_active_constraints.csv')
scenarios = ['nominal','loaded','downhill','curve','high_aoi','combined']
horizons = [1,5,10,20,40,80]
colors = ['#1864ab','#e67700','#2b8a3e','#7048e8','#c2255c','#495057']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
                     'axes.titlesize':9,'axes.labelsize':8,
                     'xtick.labelsize':7,'ytick.labelsize':7,
                     'legend.fontsize':6.5,'pdf.fonttype':42})
fig, axes = plt.subplots(2,2,figsize=(7.2,5.4),constrained_layout=True)

for scenario,color in zip(scenarios,colors):
    rr = sorted((r for r in metrics if r['scenario']==scenario),key=lambda x:int(x['H']))
    axes[0,0].plot(horizons,[float(r['qualified_fraction']) for r in rr],
                   marker='o',ms=3,color=color,label=scenario.replace('_',' '))
    axes[0,1].plot(horizons,[float(r['median_Cbar']) for r in rr],
                   marker='o',ms=3,color=color)
axes[0,0].set(xlabel='prediction horizon H (steps)',ylabel='model screen-pass fraction',
              ylim=(-.03,1.03),title='(a) Horizon-dependent screen passes')
axes[0,0].legend(ncol=2,loc='lower left')
axes[0,1].axhline(1,color='black',ls='--',lw=.7)
axes[0,1].set(xlabel='prediction horizon H (steps)',
              ylabel='median calibrated $\\bar C_H$',yscale='log',
              title='(b) Consumption bound')

values = np.zeros((len(scenarios),3))
for i,s in enumerate(scenarios):
    one = [r for r in detail if r['scenario']==s and r['H']=='1']
    by_id = {int(r['trajectory']):float(r['quality']) for r in one}
    cuts = np.quantile(list(by_id.values()),[1/3,2/3])
    allrows = [r for r in detail if r['scenario']==s and r['accepted']=='1']
    grouped = {}
    for r in allrows:
        tid = int(r['trajectory'])
        grouped.setdefault(tid,[]).append(r)
    bins = [[] for _ in range(3)]
    for tid,rs in grouped.items():
        hcert = max((int(r['H']) for r in rs if r['qualified']=='1'),default=0)
        bin_id = int(np.searchsorted(cuts,by_id[tid]))
        bins[bin_id].append(hcert)
    values[i] = [np.median(b) if b else np.nan for b in bins]
im = axes[1,0].imshow(values,cmap='viridis',vmin=0,vmax=80,aspect='auto')
axes[1,0].set(xticks=[0,1,2],xticklabels=['best quality','middle','worst quality'],
              yticks=range(len(scenarios)),
              yticklabels=[s.replace('_',' ') for s in scenarios],
              title='(c) Median descriptive CPH, steps')
for i in range(len(scenarios)):
    for j in range(3):
        axes[1,0].text(j,i,f'{values[i,j]:.0f}',ha='center',va='center',
                       color='white' if values[i,j]<48 else 'black',fontsize=7)
fig.colorbar(im,ax=axes[1,0],fraction=.042,pad=.02,label='steps')

families = ['brake_input','stopping','spacing','speed_envelope','terminal']
family_colors = ['#d9480f','#1971c2','#2f9e44','#7048e8','#868e96']
bottom = np.zeros(len(horizons))
for fam,col in zip(families,family_colors):
    y = [float(next(r['fraction'] for r in active
                    if r['scenario']=='combined' and r['H']==str(H)
                    and r['constraint']==fam)) for H in horizons]
    axes[1,1].bar(range(len(horizons)),y,bottom=bottom,color=col,
                  label=fam.replace('_',' '),width=.74)
    bottom += y
axes[1,1].set(xticks=range(len(horizons)),xticklabels=horizons,
              xlabel='prediction horizon H (steps)',ylabel='fraction of traces',
              ylim=(0,1),title='(d) Active row: combined stress')
axes[1,1].legend(ncol=2,loc='lower right')
for ax in (axes[0,0],axes[0,1],axes[1,1]):ax.grid(alpha=.15)
for ext in ('pdf','png'):
    fig.savefig(FIG/f'fig3_certifiable_horizon.{ext}',dpi=220,bbox_inches='tight')
