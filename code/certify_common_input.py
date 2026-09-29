"""Exact fixed/switching 8-s common-input lifted certificate.

For every selected load schedule e_i=L_i u_leader+E_i eps+D_i w, where the
three input families are independent and arbitrary over the 80-step horizon.
No e_i/e_(i-1) factorization, infinite-time theorem, or RPI claim is used.
"""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
from functools import lru_cache
import numpy as np
from certify import CONFIG, LOADS, GRADES, CURVES, Truck, truck, matrices, outputs, net_brake, speed_limit

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results"/"certificates";OUT.mkdir(parents=True,exist_ok=True)
H=CONFIG["horizon"];N=4
LEADER_L2_BUDGET=2.0                 # discrete norm of leader command; stress assumption
W_SAMPLE_BUDGET=0.01                 # m/s² per truck and sample; stress assumption
W_L2_BUDGET=W_SAMPLE_BUDGET*math.sqrt(N*H)
EPS_STACK_FACTOR=math.sqrt((N-1)*H)

def schedule(kind,load):
    if kind=="homogeneous": return [lambda k:"L2_medium"]+[lambda k,l=load:l for _ in range(3)]
    if kind=="heterogeneous":
        names=["L2_medium","L1_light","L2_medium","L3_loaded"]
        return [lambda k,l=l:l for l in names]
    base=["L2_medium","L1_light","L2_medium","L3_loaded"]
    funcs=[lambda k,l=l:l for l in base]
    if kind=="switch_L2_to_L3": funcs[2]=lambda k:"L2_medium" if k<H//2 else "L3_loaded"
    elif kind=="switch_L3_to_L1": funcs[3]=lambda k:"L3_loaded" if k<H//2 else "L1_light"
    elif kind=="switch_L2_L1_L3": funcs[2]=lambda k:"L2_medium" if k<H//3 else ("L1_light" if k<2*H//3 else "L3_loaded")
    else: raise ValueError(kind)
    return funcs

@lru_cache(None)
def maps(kind,load,h):
    funcs=schedule(kind,load)
    ce,cv,cg,cp=outputs(N,h)
    Su=np.zeros((3*N,H));Se=np.zeros((3*N,3*H*(N-1)));Sw=np.zeros((3*N,H*N))
    outputs_all={n:[None]+[np.zeros((H,d)) for i in range(1,N)]
                 for n,d in [("e_u",H),("e_eps",3*H*(N-1)),("e_w",H*N),
                             ("gap_u",H),("gap_eps",3*H*(N-1)),("gap_w",H*N),
                             ("v_u",H),("v_eps",3*H*(N-1)),("v_w",H*N),
                             ("pred_v_u",H),("pred_v_eps",3*H*(N-1)),("pred_v_w",H*N),
                             ("input_u",H),("input_eps",3*H*(N-1)),("input_w",H*N)]}
    sequence=[]
    for k in range(H):
        names=[fn(k) for fn in funcs];vehicles=[truck(l) for l in names]
        sequence.append(names)
        F,B0,Berr=matrices(vehicles,h)
        Su=F@Su;Su[:,k]+=B0
        Se=F@Se;Sw=F@Sw
        for i in range(1,N):
            sl=slice((i-1)*3*H+3*k,(i-1)*3*H+3*k+3)
            Se[:,sl]+=Berr[i]
        for i in range(N):Sw[3*i+2,i*H+k]+=1.0
        kp,kd,ka=(CONFIG[x] for x in ("kp","kd","ka"))
        for i in range(1,N):
            vectors={"e":ce[i-1],"gap":cg[i-1],"v":cv[i-1],"pred_v":cp[i-1]}
            cu=kp*ce[i-1]+kd*(cp[i-1]-cv[i-1]);cu[3*(i-1)+2]+=ka
            vectors["input"]=cu
            for name,c in vectors.items():
                outputs_all[name+"_u"][i][k,:]=c@Su
                outputs_all[name+"_eps"][i][k,:]=c@Se
                outputs_all[name+"_w"][i][k,:]=c@Sw
            outputs_all["input_eps"][i][k,(i-1)*3*H+3*k:(i-1)*3*H+3*k+3]+=[kp,kd,ka]
    outputs_all["sequence"]=sequence
    return outputs_all

def norm2(A):
    gram=A@A.T
    return float(math.sqrt(max(0.,np.linalg.eigvalsh(gram)[-1])))

def rownorm(A,k):return float(np.linalg.norm(A[k,:]))

def evaluate(kind,load,grade_name,curve_name,h,age):
    m=maps(kind,load,h);seq=m["sequence"]
    grades=GRADES[grade_name];curvature=CURVES[curve_name]
    gl=[norm2(m["e_u"][i]) for i in (1,2,3)]
    be=[norm2(m["e_eps"][i]) for i in (1,2,3)]
    ch=[norm2(m["e_w"][i]) for i in (1,2,3)]
    nonamp=bool(gl[1]<=gl[0]+1e-9 and gl[2]<=gl[1]+1e-9)
    candidates=[]
    v=CONFIG["speed_mps"]
    for i in (1,2,3):
        for k in range(H):
            name=seq[k][i];pred=seq[k][i-1];p=truck(name)
            bf=net_brake(name,grades);bp=net_brake(pred,grades)
            gap0=CONFIG["standstill_gap_m"]+h*v
            static={
                "spacing":gap0-CONFIG["physical_gap_min_m"]-CONFIG["model_gap_reserve_m"],
                "stopping":gap0-CONFIG["stopping_reserve_m"]-v*(age+p.tau)-v*v/(2*bf)+v*v/(2*bp)-CONFIG["model_stop_reserve_m"] if bf>0 and bp>0 else -math.inf,
                "speed_envelope":speed_limit(curvature)-v-CONFIG["model_speed_reserve_mps"],
                "brake_input":bf/p.b-CONFIG["model_brake_reserve_mps2"],
                "terminal":CONFIG["terminal_error_halfwidth_m"]-CONFIG["model_gap_reserve_m"],
            }
            def comb(name,scale):
                return tuple(m[x+"_"+f][i][k,:]*scale for f in ("u","eps","w") for x in [name])
            maps_row={"spacing":comb("gap",1),"speed_envelope":comb("v",1),
                      "brake_input":comb("input",1),"terminal":comb("e",1)}
            if bf>0 and bp>0:
                maps_row["stopping"]=tuple(m["gap_"+f][i][k,:]
                   -(age+p.tau+v/bf)*m["v_"+f][i][k,:]
                   +(v/bp)*m["pred_v_"+f][i][k,:] for f in ("u","eps","w"))
            else:maps_row["stopping"]=(np.zeros(H),np.zeros(3*H*(N-1)),np.zeros(H*N))
            for constraint,raw in static.items():
                if constraint=="terminal" and k!=H-1:continue
                lu,ee,ww=maps_row[constraint]
                leader_use=float(np.linalg.norm(lu))*LEADER_L2_BUDGET
                dyn_use=float(np.linalg.norm(ww))*W_L2_BUDGET
                margin=raw-leader_use-dyn_use
                eta=float(np.linalg.norm(ee))
                crit=margin/(eta*EPS_STACK_FACTOR) if eta>0 else math.inf
                candidates.append((crit,constraint,i,k,margin,eta,raw,leader_use,dyn_use))
    active=min(candidates,key=lambda t:t[0])
    positive=bool(active[4]>0)
    return dict(kind=kind,load=load,grade=grade_name,curvature=curve_name,
                headway_s=h,age_s=age,leader_l2_budget=LEADER_L2_BUDGET,
                disturbance_l2_budget=W_L2_BUDGET,gammaL1=gl[0],gammaL2=gl[1],gammaL3=gl[2],
                Gamma_L=max(gl),Beta=max(be),Chi=max(ch),beta1=be[0],beta2=be[1],beta3=be[2],
                chi1=ch[0],chi2=ch[1],chi3=ch[2],leader_to_link_nonamplifying=int(nonamp),
                eta_active=active[5],active_constraint=active[1],active_follower=active[2],
                active_step=active[3],nominal_margin=active[6],leader_reserve=active[7],
                disturbance_reserve=active[8],margin_after_reserves=active[4],
                epsilon_crit=active[0] if positive else float("nan"),
                feasible=int(positive),certificate_reason="common_input_finite_horizon" if positive else "nonpositive_"+active[1]+"_slack",
                horizon_steps=H,horizon_s=H*CONFIG["dt"])

def write_csv(path,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    conf={"H":H,"dt":CONFIG["dt"],"leader_l2_budget":LEADER_L2_BUDGET,
          "w_sample_budget_mps2":W_SAMPLE_BUDGET,"w_l2_budget":W_L2_BUDGET,
          "epsilon_stack_factor":EPS_STACK_FACTOR,"note":"all load/road parameters remain v2 stress assumptions"}
    (OUT/"common_input_config.json").write_text(json.dumps(conf,indent=2),encoding="utf-8")
    rows=[]
    for load in LOADS:
        for grade in GRADES:
            for curve in CURVES:
                for h in [1.,1.5,2.,2.5,3.,4.]:
                    for age in [0.,.2,.4]:rows.append(evaluate("homogeneous",load,grade,curve,h,age))
        print("homogeneous",load,len(rows),flush=True)
    for h in [1.5,2.,3.,4.]:rows.append(evaluate("heterogeneous","L3_loaded","G2_downhill","K2_moderate",h,.2))
    for kind in ["switch_L2_to_L3","switch_L3_to_L1","switch_L2_L1_L3"]:
        for h in [2.,3.,4.]:
            for age in [0.,.2,.4]:rows.append(evaluate(kind,"L3_loaded","G2_downhill","K2_moderate",h,age))
        print("switch",kind,len(rows),flush=True)
    write_csv(OUT/"common_input_certificate_table.csv",rows)
    print("rows",len(rows),"positive",sum(r["feasible"] for r in rows),
          "nonamplifying",sum(r["leader_to_link_nonamplifying"] for r in rows),flush=True)

if __name__=="__main__":main()
