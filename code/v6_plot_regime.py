"""Five-panel budget x stress regime map from paired held-out episodes."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/v6'
with (DATA/'regime_contrasts.csv').open(newline='',encoding='utf-8') as f:
    contrasts=list(csv.DictReader(f))
with (DATA/'regime_active_best.csv').open(newline='',encoding='utf-8') as f:
    active=list(csv.DictReader(f))
scenarios=['nominal','loaded','downhill','curve','high_aoi','combined']
budgets=[5.,8.,10.,12.,16.,24.]
methods=['ap','aoi','state','cr_voi_1']
mlabel=['AP','AoI','State','CR']
mcolor=['#e67700','#7048e8','#1971c2','#d6336c']
families=['brake_input','stopping','spacing','speed_envelope','terminal']
flabel=['Brake','Stop','Gap','Speed','Term.']
fcolor=['#d9480f','#1971c2','#2f9e44','#7048e8','#868e96']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,
                     'axes.titlesize':8,'axes.labelsize':7,
                     'xtick.labelsize':6.2,'ytick.labelsize':6.2,
                     'pdf.fonttype':42})
fig,ax=plt.subplots(2,3,figsize=(7.25,5.4),constrained_layout=True)
limit=max(abs(float(r['mean_delta_c'])) for r in contrasts)
limit=min(max(.08,limit),1.5)
for j,comp in enumerate(['ap','aoi','state']):
    grid=np.array([[float(next(r['mean_delta_c'] for r in contrasts
                       if r['scenario']==s and float(r['budget_kb'])==b and
                       r['comparator']==comp)) for b in budgets] for s in scenarios])
    im=ax[0,j].imshow(grid,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
    for i,s in enumerate(scenarios):
        for k,b in enumerate(budgets):
            r=next(r for r in contrasts if r['scenario']==s and
                   float(r['budget_kb'])==b and r['comparator']==comp)
            if float(r['ci95_high'])<0 or float(r['ci95_low'])>0:
                ax[0,j].plot(k,i,'o',ms=1.65,
                             color='white' if abs(grid[i,k])>.48*limit else 'black')
    ax[0,j].set(title=f'({chr(97+j)}) CR minus {["AP","AoI","State"][j]}: mean C',
                xticks=range(6),xticklabels=[int(x) for x in budgets],
                yticks=range(6),
                yticklabels=[s.replace('_',' ') for s in scenarios] if j==0 else [],
                xlabel='budget (kbit/slot)')
fig.colorbar(im,ax=ax[0,:],orientation='horizontal',shrink=.55,pad=.015,
             label='paired mean ΔC (negative favors CR)')
lookup={(r['scenario'],float(r['budget_kb'])):r for r in active}
best=np.array([[methods.index(lookup[s,b]['best_method']) for b in budgets]
               for s in scenarios])
act=np.array([[families.index(lookup[s,b]['active_constraint']) for b in budgets]
              for s in scenarios])
for axis,values,labels,colors,title in [
    (ax[1,0],best,mlabel,mcolor,'(d) Best of four by mean C'),
    (ax[1,1],act,flabel,fcolor,'(e) CR active row')]:
    cmap=ListedColormap(colors)
    axis.imshow(values,cmap=cmap,norm=BoundaryNorm(
        np.arange(-.5,len(labels)+.5,1),cmap.N),aspect='auto')
    for i in range(6):
        for j in range(6):
            axis.text(j,i,labels[values[i,j]],ha='center',va='center',
                      color='white',fontsize=5.8,fontweight='bold')
    axis.set(title=title,xticks=range(6),xticklabels=[int(x) for x in budgets],
             yticks=range(6),yticklabels=[s.replace('_',' ') for s in scenarios]
             if axis is ax[1,0] else [],xlabel='budget (kbit/slot)')
ax[1,2].axis('off')
cr_cells=int(np.sum(best==methods.index('cr_voi_1')))
sig_ap=sum(float(r['ci95_high'])<0 for r in contrasts if r['comparator']=='ap')
ax[1,2].text(.02,.9,'Reading the map',fontsize=8,fontweight='bold',
             transform=ax[1,2].transAxes)
ax[1,2].text(.02,.75,'Blue: CR has lower consumption.\nRed: comparator has lower consumption.\nDot: paired 95% seed CI excludes zero.',
             va='top',transform=ax[1,2].transAxes,fontsize=7,linespacing=1.5)
ax[1,2].text(.02,.4,f'CR best in {cr_cells}/36 cells.\nCR < AP with CI in {sig_ap}/36 cells.\nEach cell: 12 paired seeds × 80 slots.',
             va='top',transform=ax[1,2].transAxes,fontsize=7,linespacing=1.5)
fig.suptitle('Where does control relevance help? Model-based paired sweep',
             fontsize=9)
for ext in ('pdf','png'):
    fig.savefig(ROOT/f'figures/fig5_regime_map.{ext}',dpi=250,bbox_inches='tight')
