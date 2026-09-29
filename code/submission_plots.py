"""Re-render existing plot data only. Block every write to the results tree."""
from pathlib import Path
import builtins, io, sys, re
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
_open=builtins.open;_ioopen=io.open
def guarded(original):
    def call(file,mode='r',*a,**kw):
        if isinstance(file,(str,Path)) and any(x in mode for x in 'wax+'):
            p=Path(file).resolve()
            if ROOT/'results' in p.parents:
                raise RuntimeError('Immutable result write blocked: '+str(p))
        return original(file,mode,*a,**kw)
    return call
builtins.open=guarded(_open);io.open=guarded(_ioopen)
def plot(name):
    p=ROOT/'code'/name;s=p.read_text(encoding='utf-8')
    if name=='v5_plot_horizon.py':
        s=s.replace('CPH','CRIH').replace('fig3_certifiable_horizon','fig3_information_horizon')
    if name=='v5_plot_relevance.py':
        s=s.replace('Same fresh accepted matches','Same held-out accepted matches').replace('Frozen directional axes','Directional error axes')
        s=s.replace("label=['gap','relative speed','acceleration'][i]","label=['gap (m)','relative speed (m/s)','acceleration (m/s²)'][i]")
        s=s.replace("legend(ncol=3,loc='upper right')","legend(ncol=1,loc='upper right')")
    if name=='v5_plot_gsppo.py':
        s=s.replace('Certificate consumption','Constraint consumption').replace('\\\\bar C','\\\\widetilde C')
    if name=='v6_plot_pareto.py':
        start=s.index('rows=read(');end=s.index('colors=')
        s=s[:start]+'''main_methods=('ap','aoi','state','cr_voi_1')
points=read(OUT/'received_pareto_points.csv')
summary=read(OUT/'received_pareto_cell_summary.csv')
front=read(OUT/'received_pareto_frontier.csv')
for r in summary:
    for k in list(r):
        if k.endswith('_mean') or k.endswith('_halfwidth'):r[k]=float(r[k])
for r in front:
    for k in ('received_kbit_per_s','value','nondominated'):r[k]=float(r[k])
combined=[r for r in summary if r['scenario']=='combined']
'''+s[end:]
        s=s.replace('rasterized=True','rasterized=False')
        s=s.replace("if float(r['budget_kb']) in (5.,10.,24.):", "if float(r['budget_kb']) in (5.,10.,24.) and (m=='ap' or float(r['budget_kb'])!=24.):")
        s=s.replace("xytext=(2,2)","xytext=((4,6) if m=='ap' else (5,-10))")
    if name=='v6_plot_disagreement.py':
        start=s.index('if case:\n    chosen=');end=s.index('plt.rcParams.update')
        s=s[:start]+'''with (OUT/'ap_control_disagreement_summary.csv').open(newline='',encoding='utf-8') as f:
    summary=list(csv.DictReader(f))
for r in summary:
    r['fraction']=float(r['fraction'])
'''+s[end:]
        s=s.replace("'fraction'","'fraction'",1)
        s=s.replace("'(a) All paired episodes; upper-right = disagreement'","'(a) All paired episodes'")
    if name=='v6_plot_regime.py':
        s=s.replace('seed CI','seed interval').replace('AP with CI','AP with interval')
        s=s.replace("'Brake','Stop','Gap','Speed','Term.'","'B','Stop','Gap','Speed','Terminal'")
    # Preserve data and marks; improve publication-size text.
    s=re.sub(r"fontsize=(5\.5|5\.8|6|6\.2|6\.3|6\.5)(?=[,)])",'fontsize=7',s)
    s=re.sub(r"'(xtick.labelsize|ytick.labelsize|legend.fontsize)':(6|6\.2|6\.5|6\.8)(?=[,}])",r"'\1':7",s)
    if name=='v6_plot_mechanism.py':
        s=s.replace('table.set_fontsize(6.2)','table.set_fontsize(7)')
        s=s.replace('fontsize=6.8','fontsize=7').replace("'(d) Constraint consumption'","'(d) Constraint consumption'")
        s=s.replace("'model C(t)'","r'model $\\widetilde C(t)$'").replace("'model 1−C(t)'","r'model $1-\\widetilde C(t)$'")
        s=s.replace("'fraction / slot'","'indicator (0/1)'")
    exec(compile(s,str(p),'exec'),{'__file__':str(p),'__name__':'submission_render'})
    import matplotlib.pyplot as plt
    plt.close('all')
    print('Rendered',name,flush=True)

def architecture():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(7.2,2.7));ax.set(xlim=(0,12),ylim=(0,4));ax.axis('off')
    titles=['Perception error','Constraint consumption','Information horizon','Communication value']
    bodies=['Gap, relative speed,\npredecessor acceleration','Common-input row maps\nand remaining margins','Directional uncertainty\nover prediction steps','Expected restoration\nper bit / latency cost']
    cols=['#1971c2','#a61e4d','#5f3dc4','#2b8a3e']
    for i,(title,body,col) in enumerate(zip(titles,bodies,cols)):
        x=.08+i*3.04
        ax.add_patch(FancyBboxPatch((x,2.0),2.68,1.55,boxstyle='round,pad=.06',edgecolor=col,facecolor=col+'15',lw=1.1))
        ax.text(x+1.34,3.15,title,ha='center',va='center',fontsize=8,weight='bold',color=col)
        ax.text(x+1.34,2.48,body,ha='center',va='center',fontsize=8)
        if i<3:ax.add_patch(FancyArrowPatch((x+2.77,2.8),(x+2.97,2.8),arrowstyle='-|>',mutation_scale=11))
    for x,w,title,body,col in [(.1,5.45,'Empirical perception evidence','Held-out TruckV2X: measured one-update errors',cols[0]),(6.35,5.45,'Model-based temporal / closed-loop evidence','Generated temporal errors and paired network / plant traces',cols[2])]:
        ax.add_patch(FancyBboxPatch((x,.2),w,1.13,boxstyle='round,pad=.06',edgecolor=col,facecolor=col+'10',lw=1))
        ax.text(x+w/2,.99,title,ha='center',va='center',fontsize=8,weight='bold',color=col)
        ax.text(x+w/2,.5,body.replace('Generated temporal errors and paired network / plant traces','Temporal errors and paired network / plant traces'),ha='center',va='center',fontsize=7.8)
    ax.annotate('',xy=(2.8,1.39),xytext=(2.8,1.94),arrowprops=dict(arrowstyle='->',color=cols[0]))
    ax.annotate('',xy=(9.1,1.39),xytext=(9.1,1.94),arrowprops=dict(arrowstyle='->',color=cols[2]))
    fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(ROOT/f'figures/fig1_architecture.{ext}',dpi=240,bbox_inches='tight')
    plt.close(fig)

def supplement_plots():
    p=ROOT/'code/v7_figures.py';s=p.read_text(encoding='utf-8')
    # Plot helpers also export their input frames; disable those exports here.
    original=pd.DataFrame.to_csv
    pd.DataFrame.to_csv=lambda *a,**k:None
    s=s.replace("figsize=(10.5,6.5)","figsize=(7.2,5.6)").replace("figsize=(10.4,8.7)","figsize=(7.2,7.1)")
    s=s.replace("figsize=(10,9)","figsize=(7.2,6.3)").replace("figsize=(10,12)","figsize=(7.2,7.6)")
    s=s.replace('fontsize=5','fontsize=7').replace('fontsize=6','fontsize=7')
    s=s.replace("'H=0','0<H≤.55','H>.55'","'E=0','0<E≤.55','E>.55'")
    s=s.replace("'First-packet disagreement fraction'","'First-packet disagreement'")
    s=s.replace("'C  Denominator ablation, ±1.96 descriptive SE'","'C  Cost divisor: mean ±1.96 SE'")
    s=s.replace("'D  Common-state denominator components'","'D  Divisor and packet ranking'")
    s=s.replace('All cells retain 12 paired seeds.','Each scenario retains 12 paired seeds.')
    s=s.replace("LABEL[list(LABEL)[int(mat[i,j])]]", "{'ap':'AP','aoi':'Age','state':'St','cr_delta_c':'DC','cr_voi':'VoI'}[list(LABEL)[int(mat[i,j])]]")
    s=s.replace("f'{value:+.2f}'", "f'{value:.2f}'")
    s=s.replace("'Negative contrast favors CR.\\nEach scenario retains 12 paired seeds.\\nRow diversity uses a policy-neutral reference.'", "'Negative contrast favors CR.\\nEach scenario: 12 paired seeds.\\nPolicy-neutral reference diversity.\\n\\nSt = State; Age = AoI\\nDC = CR-DeltaC; VoI = CR-VoI'")
    s=s.replace('Automatically selected switching trace: {name}, seed {seed}, cap 10','Selected switching trace: {name}, seed {seed}, cap 10 kbit/slot')
    ns={'__file__':str(p),'__name__':'submission_render'}
    exec(compile(s,str(p),'exec'),ns)
    out=ROOT/'results/v7'
    ns['fig7'](pd.read_csv(out/'paired_contrasts.csv'))
    ns['fig8'](pd.read_csv(out/'paired_contrasts.csv'))
    ns['fig9'](pd.read_csv(out/'episode_metrics.csv'),pd.read_csv(out/'paired_contrasts.csv'),pd.read_csv(out/'ranking_disagreement.csv'))
    ns['fig10'](pd.read_csv(out/'reference_diversity.csv'),pd.read_csv(out/'slot_metrics.csv'))
    pd.DataFrame.to_csv=original
    print('Rendered four supplementary ablation figures',flush=True)
if __name__=='__main__':
    architecture()
    for name in ['v5_plot_relevance.py','v5_plot_horizon.py','v6_plot_pareto.py','v6_plot_regime.py','v6_plot_mechanism.py','v6_plot_disagreement.py','v5_plot_gsppo.py']:plot(name)
    supplement_plots()
