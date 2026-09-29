"""Empirical v4 audit + frozen control geometry in a two-column figure."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/'figures'
def read(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
rows=[r for r in read(ROOT/'results/calibration/fresh_certificate_comparison.csv')
      if r['operating_condition']=='light_brake']
accept=[r for r in rows if int(r['accepted'])]
metrics=[r for r in read(ROOT/'results/calibration/fresh_comparison_metrics.csv')
         if r['condition']=='light_brake' and r['population']=='accepted']
axes_rows=read(ROOT/'results/certificates/directional_axes_517.csv')
chosen=[]
for load,grade,curve,h,age in [
    ('L1_light','G1_level','K1_straight',2.,.2),
    ('L1_light','G1_level','K1_straight',1.,.4),
    ('L3_loaded','G2_downhill','K2_moderate',3.,.2)]:
    chosen.append(next(r for r in axes_rows if r['load']==load and
        r['grade']==grade and r['curvature']==curve and
        float(r['headway_s'])==h and float(r['age_s'])==age))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
                     'axes.titlesize':9,'axes.labelsize':8,
                     'xtick.labelsize':7,'ytick.labelsize':7,
                     'legend.fontsize':6.5,'pdf.fonttype':42})
fig,ax=plt.subplots(2,2,figsize=(7.2,5.1),constrained_layout=True)
norm=np.array([float(r['state_norm']) for r in rows])
c=np.array([float(r['C_true']) for r in rows])
ax[0,0].scatter(norm,c,s=8,alpha=.48,
    c=['#2b8a3e' if int(r['accepted']) else '#adb5bd' for r in rows])
ax[0,0].axhline(1,color='#c92a2a',ls='--',lw=.8)
ax[0,0].set(xscale='log',yscale='log',xlabel='standardized state-error norm',
            ylabel='actual one-update consumption',
            title='(a) Related, but not identical risks')
methods=['U1_ball','U2_box','U3_ellipsoid','U4_projected']
label=['ball','box','ellipsoid','projected']
med=[float(next(r['median_bound'] for r in metrics if r['method']==m))
     for m in methods]
cov=[float(next(r['coverage'] for r in metrics if r['method']==m))
     for m in methods]
x=np.arange(4)
ax[0,1].bar(x,med,color=['#74c0fc','#ffa94d','#b197fc','#51cf66'])
ax[0,1].axhline(1,color='#c92a2a',ls='--',lw=.8)
for i,y in enumerate(med):ax[0,1].text(i,y+.06,f'{y:.2f}',ha='center',fontsize=7)
ax[0,1].set(xticks=x,xticklabels=label,ylabel='median upper consumption',
            title='(b) Same fresh accepted matches')
ax2=ax[0,1].twinx()
ax2.plot(x,cov,color='#343a40',marker='o',lw=1,ms=3)
ax2.set(ylabel='empirical coverage',ylim=(.90,1.01))
width=.23
for i,(key,col) in enumerate(zip(('q_gap_crit','q_vrel_crit','q_apred_crit'),
                                  ('#1971c2','#e67700','#7048e8'))):
    ax[1,0].bar(np.arange(3)+(i-1)*width,[float(r[key]) for r in chosen],
                width,color=col,label=['gap','relative speed','acceleration'][i])
ax[1,0].set(xticks=range(3),xticklabels=['light brake','light stopping','loaded downhill'],
            ylabel='one-channel allowable bound (native units)',
            title='(c) Frozen directional axes')
ax[1,0].legend(ncol=3,loc='upper right')
counts=[sum(r['dominant_absolute_channel']==k for r in rows)
        for k in ('gap','relative_velocity','acceleration')]
ax[1,1].bar(['gap','relative speed','acceleration'],np.array(counts)/len(rows),
            color=['#1971c2','#e67700','#7048e8'])
for i,n in enumerate(counts):ax[1,1].text(i,n/len(rows)+.015,str(n),
                                      ha='center',fontsize=7)
ax[1,1].set(ylabel='fraction of 325 matched updates',ylim=(0,.78),
            title='(d) Dominant actual constraint contribution')
for a in ax.ravel():a.grid(alpha=.14)
for ext in ('pdf','png'):
    fig.savefig(FIG/f'fig2_control_relevance.{ext}',dpi=220,bbox_inches='tight')
