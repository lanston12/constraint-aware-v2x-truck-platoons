"""Closed-loop mechanism figure with automatically selected candidate insets."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/v6'
def read(name):
    with (OUT/name).open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))
records=read('mechanism_trace.csv')
candidates=read('mechanism_candidates.csv')
events=read('mechanism_selected_events.csv')
methods=['ap','aoi','cr_voi_1']
names=['AP','AoI','CR']
colors=['#e67700','#7048e8','#d6336c']
data={m:sorted([r for r in records if r['method']==m],
               key=lambda r:float(r['time_s'])) for m in methods}
t=np.array([float(r['time_s']) for r in data['ap']])
if any(len(data[m])!=80 for m in methods):raise ValueError('Missing aligned trace')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,
                     'axes.titlesize':8,'axes.labelsize':6.8,
                     'xtick.labelsize':6,'ytick.labelsize':6,
                     'legend.fontsize':6.5,'pdf.fonttype':42})
fig=plt.figure(figsize=(7.25,8.1))
grid=fig.add_gridspec(5,2,height_ratios=[1,1,1,1,1.75],
                      left=.08,right=.94,top=.93,bottom=.045,
                      hspace=.52,wspace=.30)
ax=np.array([[fig.add_subplot(grid[i,j]) for j in range(2)]
             for i in range(4)])
a=ax[0,0];a.plot(t,[float(r['snr_dB']) for r in data['ap']],
                 color='#495057',lw=1.1,label='SNR')
a2=a.twinx()
for m,n,c in zip(methods,names,colors):
    a2.plot(t,[float(r['aoi_s']) for r in data[m]],color=c,lw=.85,alpha=.83)
a.set(title='(a) SNR and AoI',ylabel='SNR (dB)')
a2.set_ylabel('AoI (s)',fontsize=6.5)
a=ax[0,1]
for m,n,c in zip(methods,names,colors):
    a.step(t,[int(r['receiver'].split('|')[0]) if r['receiver'] else 0
              for r in data[m]],where='post',color=c,lw=.9,label=n)
a.set(title='(b) First selected receiver',ylabel='receiver ID',
      yticks=[0,1,2,3])
a.legend(ncol=3,loc='upper right')
a=ax[1,0]
families=['brake_input','stopping','spacing','speed_envelope','terminal']
family_colors=['#d9480f','#1971c2','#2f9e44','#7048e8','#868e96']
families_grid=np.array([[families.index(r['active_constraint'])
    for r in data[m]] for m in methods])
a.imshow(families_grid,aspect='auto',interpolation='nearest',
         extent=(t[0],t[-1],2.5,-.5),
         cmap=ListedColormap(family_colors),vmin=-.5,vmax=4.5)
a.set(title='(c) Active control row',yticks=[0,1,2],yticklabels=names)
a.text(4.1,1,'brake-input row throughout',ha='center',va='center',
       color='white',fontsize=6.8,fontweight='bold')
for a,field,title,ylabel in [
    (ax[1,1],'c_after','(d) Constraint consumption','model C(t)'),
    (ax[2,0],'active_constraint_margin','(e) Active margin','model 1−C(t)'),
    (ax[2,1],'delivered_bits_kb','(f) Received bits','kbit / slot'),
    (ax[3,0],'fallback','(g) No packet delivered','fraction / slot')]:
    for m,n,c in zip(methods,names,colors):
        vals=[float(r[field]) for r in data[m]]
        if field in ('delivered_bits_kb','fallback'):
            a.step(t,vals,where='post',color=c,lw=.85)
        else:a.plot(t,vals,color=c,lw=1.05)
    a.set(title=title,ylabel=ylabel)
ax[1,1].axhline(1,color='black',ls='--',lw=.55)
ax[2,0].axhline(0,color='black',ls='--',lw=.55)
ax[3,0].set_ylim(-.1,1.1)
a=ax[3,1]
for m,n,c in zip(methods,names,colors):
    a.plot(t,[float(r['spacing_rmse_step_m']) for r in data[m]],
           color=c,lw=.8)
a.set(title='(h) Spacing / command (context)',ylabel='spacing RMSE (m)')
cmd=ax[3,1].twinx()
cmd.plot(t,[float(r['brake_command_mps2']) for r in data['cr_voi_1']],
         color='#495057',ls=':',lw=.8)
cmd.set_ylabel('command (m/s²)',fontsize=6.5)
for axis in ax.ravel():
    axis.set_xlim(.1,8)
    axis.grid(alpha=.12)
    if axis is not ax[1,0]:
        for j,event in enumerate(events):
            axis.axvline(float(event['time_s']),ls='--',lw=.75,
                         color='#212529',alpha=.55)
for axis in ax[-1]:axis.set_xlabel('time (s)')
table_ax=fig.add_subplot(grid[4,:])
table_ax.axis('off')
table_ax.text(0,.99,'Automatically selected decision forks',
              transform=table_ax.transAxes,fontsize=8,fontweight='bold',
              va='top')
table_ax.text(0,.86,'First packet scored on the same CR pre-decision state; '
              'S/F = state/feature, N/M/H = compression. Expected single-packet values.',
              transform=table_ax.transAxes,fontsize=6.3,va='top')
selected_steps={r['step'] for r in events}
selected=sorted([r for r in candidates if r['step'] in selected_steps],
                key=lambda r:(int(r['step']),methods.index(r['method'])))
headers=['t / policy','candidate','ΔAP','ΔAoI s','Δ||σ||','ΔC','kbit','ms','active row']
cells=[]
for r in selected:
    cells.append([f"{float(r['time_s']):.1f} {dict(zip(methods,names))[r['method']]}",
                  r['candidate_packet'],
                  f"{float(r['expected_ap_proxy_gain']):.3f}",
                  f"{float(r['expected_aoi_reduction_s']):.3f}",
                  f"{float(r['expected_state_norm_reduction']):.3f}",
                  f"{float(r['expected_delta_c']):.3f}",
                  f"{float(r['bits_kb']):.0f}",
                  f"{float(r['latency_ms']):.1f}",
                  r['active_row_after'].replace('_input','')])
table=table_ax.table(cellText=cells,colLabels=headers,cellLoc='center',
                     colWidths=[.12,.20,.09,.09,.10,.08,.07,.07,.12],
                     bbox=[0,0,.98,.76])
table.auto_set_font_size(False)
table.set_fontsize(6.2)
for (i,j),cell in table.get_celld().items():
    cell.set_linewidth(.35)
    cell.set_edgecolor('#adb5bd')
    if i==0:
        cell.set_facecolor('#e9ecef')
        cell.set_text_props(weight='bold')
    elif i in (1,4):
        cell.set_facecolor('#f8f9fa')
fig.suptitle('combined heavy-duty stress: packet choice to control margin',
             fontsize=9.5,y=.98)
for ext in ('pdf','png'):
    fig.savefig(ROOT/f'figures/fig6_combined_mechanism.{ext}',
                dpi=260,bbox_inches='tight')
