"""Algebraic consistency check of the frozen map, not an experiment."""
from pathlib import Path
import json, math
import numpy as np
from certify import CONFIG, truck
from certify_common_input import maps, schedule, H
ROOT=Path(__file__).resolve().parents[1]
kp,kd,ka=[CONFIG[x] for x in ('kp','kd','ka')]
sel=np.eye(12)
checks=[]
for kind in ('homogeneous','heterogeneous','switch_L2_to_L3','switch_L3_to_L1','switch_L2_L1_L3'):
    load='L3_loaded';h=3.;functions=schedule(kind,load);ref=maps(kind,load,h)
    S=np.zeros((12,14*H));error=0.
    for k in range(H):
        A=np.zeros((12,12));Bu=np.zeros(12);Br=np.zeros((12,9));Bw=np.zeros((12,4))
        for i in range(4):
            p=truck(functions[i](k));alpha=math.exp(-.1/p.tau);beta=(1-alpha)*p.b
            A[3*i]=sel[3*i]+.1*sel[3*i+1]
            A[3*i+1]=sel[3*i+1]+.1*sel[3*i+2]
            A[3*i+2]=alpha*sel[3*i+2]
            Bw[3*i+2,i]=1
            if i==0:Bu[2]=1-alpha
            else:
                ce=sel[3*(i-1)]-sel[3*i]-h*sel[3*i+1]
                cu=kp*ce+kd*(sel[3*(i-1)+1]-sel[3*i+1])+ka*sel[3*(i-1)+2]
                A[3*i+2]+=beta*cu
                Br[3*i+2,3*(i-1):3*i]=beta*np.array([kp,kd,ka])
        S=A@S;S[:,k]+=Bu
        for i in range(1,4):S[:,H+(i-1)*3*H+3*k:H+(i-1)*3*H+3*k+3]+=Br[:,3*(i-1):3*i]
        for i in range(4):S[:,10*H+i*H+k]+=Bw[:,i]
        for i in range(1,4):
            ce=sel[3*(i-1)]-sel[3*i]-h*sel[3*i+1]
            cu=kp*ce+kd*(sel[3*(i-1)+1]-sel[3*i+1])+ka*sel[3*(i-1)+2]
            rows={'e':ce,'gap':sel[3*(i-1)]-sel[3*i],'v':sel[3*i+1],'pred_v':sel[3*(i-1)+1],'input':cu}
            for name,c in rows.items():
                out=c@S
                if name=='input':out[H+(i-1)*3*H+3*k:H+(i-1)*3*H+3*k+3]+=[kp,kd,ka]
                for fam,sl in [('u',slice(0,H)),('eps',slice(H,10*H)),('w',slice(10*H,14*H))]:
                    error=max(error,float(np.max(np.abs(out[sl]-ref[name+'_'+fam][i][k]))))
    assert error<1e-12,(kind,error)
    checks.append({'load_schedule':kind,'maximum_absolute_map_difference':error,'status':'PASS'})
(ROOT/'audit/lift_consistency.json').write_text(json.dumps({'type':'algebraic map check only; no new research trajectories','checks':checks,'brake_row_timing':'c_u X(k+1) plus direct r_i(k); causal controller uses X(k)','initial_perturbation':'zero','leader_gain':1},indent=2),encoding='utf-8')
print(json.dumps(checks))
