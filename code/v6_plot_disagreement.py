"""All paired AP-vs-CR episodes; deterministic maximum-disagreement exemplar."""
import csv
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/v6'
with (OUT/'ap_control_disagreement.csv').open(newline='',encoding='utf-8') as f:
    rows=list(csv.DictReader(f))
for r in rows:
    r['x']=float(r['ap_proxy_advantage_ap'])
    r['y']=float(r['control_margin_advantage_cr'])
    r['score']=max(r['x'],0)*max(r['y'],0)
eligible=[r for r in rows if r['x']>0 and r['y']>0]
case=sorted(eligible,key=lambda r:(-r['score'],r['scenario'],
                                   float(r['budget_kb']),int(r['seed'])))[0] if eligible else None
if case:
    chosen=[dict(scenario=case['scenario'],budget_kb=case['budget_kb'],
                 seed=case['seed'],selection_rule='max positive x*y among all paired episodes',
                 ap_proxy_advantage_ap=case['x'],
                 control_margin_advantage_cr=case['y'],
                 disagreement_score=case['score'])]
    with (OUT/'ap_control_representative.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=chosen[0]);w.writeheader();w.writerows(chosen)
summary=[]
for axis,values in [('all',['all']),
                    ('budget',sorted({r['budget_kb'] for r in rows},key=float)),
                    ('stress',sorted({r['scenario'] for r in rows}))]:
    for value in values:
        subset=rows if axis=='all' else [r for r in rows if
            (r['budget_kb'] if axis=='budget' else r['scenario'])==value]
        count=sum(r['x']>0 and r['y']>0 for r in subset)
        summary.append(dict(axis=axis,value=value,n=len(subset),
                            disagreement_count=count,fraction=count/len(subset)))
with (OUT/'ap_control_disagreement_summary.csv').open(
        'w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=summary[0]);w.writeheader();w.writerows(summary)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
                     'axes.titlesize':9,'axes.labelsize':8,
                     'xtick.labelsize':7,'ytick.labelsize':7,
                     'pdf.fonttype':42})
fig=plt.figure(figsize=(7.25,4.4),constrained_layout=True)
gs=fig.add_gridspec(2,2,width_ratios=(1.5,1))
a=fig.add_subplot(gs[:,0])
palette={'nominal':'#1971c2','loaded':'#d9480f','downhill':'#2b8a3e',
         'curve':'#7048e8','high_aoi':'#c2255c','combined':'#495057'}
for s,c in palette.items():
    sub=[r for r in rows if r['scenario']==s]
    a.scatter([r['x'] for r in sub],[r['y'] for r in sub],s=14,
              alpha=.53,label=s.replace('_',' '),color=c)
a.axhline(0,color='#495057',lw=.7);a.axvline(0,color='#495057',lw=.7)
a.axvspan(0,max(r['x'] for r in rows)*1.05,ymin=.5,ymax=1,
          color='#51cf66',alpha=.08,zorder=-10)
if case:
    a.scatter([case['x']],[case['y']],s=80,facecolors='none',
              edgecolors='black',lw=1.3,zorder=20)
    a.annotate(f"max disagreement:\n{case['scenario']}, {case['budget_kb']} kbit, seed {case['seed']}",
               (case['x'],case['y']),xytext=(-105,20),textcoords='offset points',
               arrowprops={'arrowstyle':'->','lw':.7},fontsize=7)
a.set(xlabel='AP proxy advantage: AP − CR',
      ylabel='minimum model margin advantage: CR − AP',
      title='(a) All paired episodes; upper-right = disagreement')
a.grid(alpha=.15)
a.legend(ncol=2,loc='lower right',fontsize=6)
for axis,kind,title in [(fig.add_subplot(gs[0,1]),'budget','(b) By budget'),
                        (fig.add_subplot(gs[1,1]),'stress','(c) By stress')]:
    rr=[r for r in summary if r['axis']==kind]
    labels=[r['value'].replace('_',' ') for r in rr]
    vals=[r['fraction'] for r in rr]
    axis.bar(range(len(rr)),vals,color='#1971c2')
    axis.set(xticks=range(len(rr)),xticklabels=labels,ylim=(0,max(.3,max(vals)*1.2)),
             ylabel='fraction',title=title)
    if kind=='stress':axis.tick_params(axis='x',rotation=35)
    for i,r in enumerate(rr):
        axis.text(i,vals[i]+.008,f"{r['disagreement_count']}/{r['n']}",
                  ha='center',fontsize=6)
    axis.grid(axis='y',alpha=.15)
fig.suptitle('AP–control disagreement from model-based paired scheduling',fontsize=9)
for ext in ('pdf','png'):
    fig.savefig(ROOT/f'figures/fig_s1_ap_control_disagreement.{ext}',
                dpi=250,bbox_inches='tight')
