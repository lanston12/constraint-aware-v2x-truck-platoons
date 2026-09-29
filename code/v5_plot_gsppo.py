"""Held-out comparison of three paper-inspired PPO variants."""
import csv
from pathlib import Path
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def read(name):
    with (ROOT/'results/v5'/name).open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))
history=read('gsppo_training_history.csv')
allrows=read('gsppo_heldout_metrics.csv')
rows=[r for r in allrows if r['scenario']=='combined' and
      float(r['budget_kb'])==10.]
methods=['fan_style','cr_voi','shielded_cr_voi']
labels=['Fan-style PPO','CR-VoI PPO','Shielded CR-VoI']
colors=['#e67700','#1971c2','#d6336c']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
                     'axes.titlesize':9,'axes.labelsize':8,
                     'xtick.labelsize':7,'ytick.labelsize':7,
                     'legend.fontsize':7,'pdf.fonttype':42})
fig,ax=plt.subplots(2,3,figsize=(7.2,5.0),constrained_layout=True)
for method,label,col in zip(methods,labels,colors):
    rr=sorted((r for r in history if r['variant']==method),
              key=lambda x:int(x['update']))
    y=np.array([float(r['train_mean_c']) for r in rr])
    smooth=np.convolve(y,np.ones(5)/5,mode='valid')
    ax[0,0].plot(range(5,len(y)+1),smooth,label=label,color=col,lw=1.2)
ax[0,0].set(xlabel='PPO update',ylabel='training mean model $\\bar C$',
            title='(a) Training convergence')
ax[0,0].legend()

def means_ci(field):
    means=[];ci=[]
    for method in methods:
        x=np.array([float(r[field]) for r in rows if r['method']==method])
        means.append(x.mean())
        ci.append(t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)))
    return np.array(means),np.array(ci)
fields=[('mean_ap_proxy','AP proxy (unitless)','(b) Perception proxy'),
        ('received_kbit_per_s','received kbit/s','(c) Delivered communication'),
        ('mean_c','mean model $\\bar C$','(d) Certificate consumption'),
        ('minimum_active_constraint_margin','minimum $1-\\bar C$',
         '(e) Worst model margin'),
        ('spacing_rmse_m','spacing RMSE (m)','(f) Closed-loop spacing')]
for axis,(field,ylabel,title) in zip([ax[0,1],ax[0,2],
                                      ax[1,0],ax[1,1],ax[1,2]],fields):
    y,e=means_ci(field)
    axis.bar(range(3),y,yerr=e,capsize=2,color=colors,width=.68)
    axis.set(xticks=range(3),xticklabels=['Fan','CR','Shielded'],
             ylabel=ylabel,title=title)
    axis.grid(axis='y',alpha=.15)
fig.suptitle('Model-based PPO; combined stress, 10 kbit/slot, 12 held-out paired seeds',
             fontsize=9)
for ext in ('pdf','png'):
    fig.savefig(ROOT/f'figures/fig5_gsppo_comparison.{ext}',
                dpi=220,bbox_inches='tight')
