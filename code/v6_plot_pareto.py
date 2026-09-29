"""Actual-received-bit Pareto plots; preserve policy-dependent throughput."""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/v6'
def read(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
rows=read(OUT/'regime_episode_metrics.csv')
main_methods=('ap','aoi','state','cr_voi_1')
extras=[r for r in read(ROOT/'results/v5/deterministic_episode_metrics.csv')
        if r['method'] in ('random','nearest','full_broadcast')]
points=[]
for source,collection in [('v6_full_sweep',rows),('v5_extra_baseline',extras)]:
    for r in collection:
        points.append(dict(source=source,scenario=r['scenario'],
            budget_kb=r['budget_kb'],seed=r['seed'],method=r['method'],
            received_kbit_per_s=r['received_kbit_per_s'],
            mean_c=r['mean_c'],
            minimum_active_constraint_margin=r['minimum_active_constraint_margin'],
            mean_ap_proxy=r['mean_ap_proxy'],
            spacing_rmse_m=r['spacing_rmse_m']))
with (OUT/'received_pareto_points.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=points[0]);w.writeheader();w.writerows(points)
group=defaultdict(list)
for r in points:
    group[r['scenario'],float(r['budget_kb']),r['method']].append(r)
summary=[]
for (scenario,budget,method),rr in sorted(group.items()):
    row=dict(scenario=scenario,budget_kb=budget,method=method,n=len(rr))
    for field in ('received_kbit_per_s','mean_c','minimum_active_constraint_margin',
                  'mean_ap_proxy','spacing_rmse_m'):
        x=np.array([float(r[field]) for r in rr])
        half=t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))
        row[field+'_mean']=float(x.mean())
        row[field+'_ci95_halfwidth']=float(half)
    summary.append(row)
with (OUT/'received_pareto_cell_summary.csv').open(
        'w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=summary[0]);w.writeheader();w.writerows(summary)
combined=[r for r in summary if r['scenario']=='combined']
front=[]
for r in combined:
    rate=r['received_kbit_per_s_mean']
    for obj,sign in [('mean_c',1),('minimum_active_constraint_margin',-1),
                     ('mean_ap_proxy',-1)]:
        value=sign*r[obj+'_mean']
        dominated=any(q is not r and
            q['received_kbit_per_s_mean']<=rate+1e-12 and
            sign*q[obj+'_mean']<=value+1e-12 and
            (q['received_kbit_per_s_mean']<rate-1e-12 or
             sign*q[obj+'_mean']<value-1e-12) for q in combined)
        front.append(dict(scenario='combined',metric=obj,method=r['method'],
            budget_kb=r['budget_kb'],received_kbit_per_s=rate,
            value=r[obj+'_mean'],nondominated=int(not dominated)))
with (OUT/'received_pareto_frontier.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=front[0]);w.writeheader();w.writerows(front)

colors={'ap':'#e67700','aoi':'#7048e8','state':'#1971c2','cr_voi_1':'#d6336c'}
names={'ap':'AP','aoi':'AoI','state':'State','cr_voi_1':'CR-VoI'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.2,
                     'axes.titlesize':8,'axes.labelsize':7.2,
                     'xtick.labelsize':6.5,'ytick.labelsize':6.5,
                     'legend.fontsize':6.8,'pdf.fonttype':42})
fig,ax=plt.subplots(2,2,figsize=(7.25,5.7),constrained_layout=True)
panels=[('received_kbit_per_s','mean_c','(a) Consumption','model mean C'),
        ('received_kbit_per_s','minimum_active_constraint_margin',
         '(b) Worst modeled margin','minimum 1−C'),
        ('received_kbit_per_s','mean_ap_proxy','(c) Perception proxy',
         'AP proxy (unitless)'),
        ('mean_ap_proxy','minimum_active_constraint_margin',
         '(d) AP–control relation','minimum 1−C')]
for axis,(xkey,ykey,title,ylabel) in zip(ax.ravel(),panels):
    other=[r for r in points if r['scenario']!='combined']
    axis.scatter([float(r[xkey]) for r in other],
                 [float(r[ykey]) for r in other],
                 s=3,color='#868e96',alpha=.10,rasterized=True)
    extra=[r for r in points if r['scenario']=='combined' and
           r['method'] not in main_methods]
    axis.scatter([float(r[xkey]) for r in extra],
                 [float(r[ykey]) for r in extra],
                 s=6,color='#868e96',alpha=.28,marker='x',rasterized=True)
    for m in main_methods:
        raw=[r for r in points if r['scenario']=='combined' and r['method']==m]
        sr=sorted((r for r in combined if r['method']==m),
                  key=lambda r:float(r['budget_kb']))
        axis.scatter([float(r[xkey]) for r in raw],
                     [float(r[ykey]) for r in raw],
                     s=8,color=colors[m],alpha=.14,rasterized=True)
        xs=[r[xkey+'_mean'] for r in sr]
        ys=[r[ykey+'_mean'] for r in sr]
        axis.plot(xs,ys,color=colors[m],marker='o',ms=3,lw=1.3,
                  label=names[m])
        if m in ('ap','cr_voi_1'):
            axis.errorbar(xs,ys,
                xerr=[r[xkey+'_ci95_halfwidth'] for r in sr],
                yerr=[r[ykey+'_ci95_halfwidth'] for r in sr],
                fmt='none',ecolor=colors[m],elinewidth=.6,
                capsize=1.4,alpha=.5)
        if m in ('ap','cr_voi_1'):
            for r,x,y in zip(sr,xs,ys):
                if float(r['budget_kb']) in (5.,10.,24.):
                    axis.annotate(str(int(float(r['budget_kb']))),
                                  (x,y),xytext=(2,2),textcoords='offset points',
                                  fontsize=5.5,color=colors[m])
    if xkey=='received_kbit_per_s':
        frontier=[r for r in front if r['metric']==ykey and
                  r['nondominated']==1]
        axis.scatter([r['received_kbit_per_s'] for r in frontier],
                     [r['value'] for r in frontier],s=38,
                     facecolors='none',edgecolors='#212529',
                     linewidths=.65,zorder=8)
    axis.set(title=title,xlabel=('received kbit/s' if xkey=='received_kbit_per_s'
                                  else 'AP proxy (unitless)'),ylabel=ylabel)
    axis.grid(alpha=.15)
ax[0,0].legend(ncol=2,loc='upper right')
fig.suptitle('Actual received-bit Pareto; outlined combined means are nondominated',
             fontsize=9)
for ext in ('pdf','png'):
    fig.savefig(ROOT/f'figures/fig4_received_bit_pareto.{ext}',
                dpi=250,bbox_inches='tight')
