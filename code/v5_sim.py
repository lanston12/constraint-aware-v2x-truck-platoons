"""Paired, model-based perception--network--platoon simulator.

The control row map comes from the frozen v4 lift. The error generator,
network, AP proxy and physical plant traces are explicitly synthetic. All
policies see the same candidate actions, budget, controller and seeded traces.
"""
import csv
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import numpy as np
from certify import CONFIG, GRADES, CURVES, net_brake, speed_limit, truck
from v5_horizon import fit_reference_scale, condition_map, SCENARIOS, write_csv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/v5'
DT = .1
STEPS = 80
BUDGETS = (5., 10., 16., 24.)  # kilobits per 0.1-s slot
SEEDS = tuple(range(201, 213))
METHODS = ('random','nearest','ap','aoi','state','cr_voi_1',
           'cr_voi_10','full_broadcast')
PACKETS = ('state','feature')
COMPRESSIONS = ('none','medium','high')
LINKS = ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))

@dataclass(frozen=True)
class Action:
    id: int
    source: int
    receiver: int
    packet: str
    compression: str
    channel: int
    bits_kb: float

def action_space():
    actions = [Action(0,-1,-1,'skip','none',0,0.)]
    for source,receiver in LINKS:
        for packet in PACKETS:
            for ci,compression in enumerate(COMPRESSIONS):
                bits = ([8.,5.,3.] if packet=='state' else [24.,16.,10.])[ci]
                for channel in (0,1):
                    actions.append(Action(len(actions),source,receiver,packet,
                                          compression,channel,bits))
    return tuple(actions)
ACTIONS = action_space()
BASE_STD = fit_reference_scale()[2]

@lru_cache(None)
def row_sensitivities(name):
    s=next(x for x in SCENARIOS if x['name']==name)
    F,M,step,_ = condition_map(s)
    P = np.hstack([F[:,i*240:i*240+3] for i in range(3)])/M[:,None]
    # A horizon-aware, diagonal innovation support approximation. It is an
    # algorithmic score, not the trajectory-level conformal Cbar_H.
    weights = []
    for H in (1,10):
        rows = step < H
        A = F[rows].reshape(np.sum(rows),3,80,3)[:,:,:H,:]
        discount = .75**np.arange(H)
        G = np.sqrt(np.sum((A*discount[None,None,:,None])**2,axis=2))
        G = G.reshape(np.sum(rows),9)/M[rows,None]
        weights.append(G)
    return P, tuple(weights)

def trace(seed, scenario):
    # v6 extends the same generator to the four previously unused frozen
    # condition bins. Nominal and combined retain their v5 trace identities.
    trace_bins = ('nominal','combined','loaded','downhill','curve','high_aoi')
    rng = np.random.default_rng(seed + 10000*trace_bins.index(scenario))
    shock = rng.standard_t(6,size=(STEPS,3,3))/math.sqrt(1.5)
    snr = rng.normal(17.,2.5,size=(STEPS,2))
    occ = rng.random((STEPS,3)) < .06
    if scenario=='combined':
        snr[25:52] -= 9.
        occ[18:58] |= rng.random((40,3)) < .42
    delivery = rng.random((STEPS,len(ACTIONS)))
    return dict(shock=shock,snr=snr,occ=occ,delivery=delivery)

class NetworkPlatoonEnv:
    def __init__(self, scenario='nominal', budget_kb=10., seed=101):
        self.scenario_name = scenario
        self.scenario = next(s for s in SCENARIOS if s['name']==scenario)
        self.budget = float(budget_kb)
        self.seed = seed
        self.trace = trace(seed,scenario)
        self.base = BASE_STD
        self.P, self.G = row_sensitivities(scenario)
        self.P2 = self.P**2
        self.G2 = tuple(g**2 for g in self.G)
        self.params = truck(self.scenario['load'])
        self.h = self.scenario['headway_s']
        speed = min(20.,speed_limit(CURVES[self.scenario['curvature']])-1.)
        target = CONFIG['standstill_gap_m']+self.h*speed
        self.x = np.array([0.,-target,-2*target,-3*target])
        self.v = np.full(4,speed)
        self.a = np.zeros(4)
        self.sigma = np.tile(self.base*self.scenario['uncertainty_factor'],(3,1))
        self.error = np.zeros((3,3))
        self.aoi = np.full(3,self.scenario['age_s'])
        self.k = 0
        self.ready = False
        self.rows = []

    def cbar(self,sigma,horizon=1):
        G2 = self.G2[0 if horizon==1 else 1]
        return float(1.96*np.max(np.sqrt(G2@sigma.reshape(9)**2)))

    def ctrue(self,error):
        return float(np.max(np.abs(self.P@error.reshape(9))))

    def ap_proxy(self,sigma):
        ratio = sigma/self.base[None,:]
        return float(1/(1+.28*np.mean(ratio[:,:2]**2)))

    def prepare(self):
        if self.ready: return
        k = self.k
        snr = self.trace['snr'][k]
        occ = self.trace['occ'][k]
        if self.scenario_name=='combined' and 27<=k<50:
            self.aoi += .15
        self.aoi += DT
        quality = (1+.032*np.maximum(17.-snr.mean(),0)) * (
            1+.45*occ.astype(float)) * (1+.28*self.aoi)
        target = self.base[None,:]*self.scenario['uncertainty_factor']*quality[:,None]
        self.sigma = .75*self.sigma+.25*target
        self.error = .75*self.error+math.sqrt(1-.75**2)*self.sigma*(
            self.trace['shock'][k])
        self.ready = True

    def observation(self):
        self.prepare()
        gap = self.x[:-1]-self.x[1:]
        return np.r_[self.k/STEPS,(gap-(8+self.h*self.v[1:]))/10,
            self.v/25,self.a/5,(self.sigma/self.base).reshape(9)/4,
            self.aoi/2,self.trace['snr'][self.k]/25,
            self.trace['occ'][self.k].astype(float),
            self.budget/24,
            float(self.scenario_name=='combined')].astype(np.float32)

    def action_mask(self):
        return np.array([a.bits_kb<=self.budget for a in ACTIONS],bool)

    def action_model(self,a,sigma=None):
        if sigma is None:sigma=self.sigma
        if a.id==0:return sigma.copy(),0.,0.
        snr = self.trace['snr'][self.k,a.channel]+(1.5 if a.channel else 0.)
        compression_level = COMPRESSIONS.index(a.compression)
        latency = .006+a.bits_kb/max(.35,1.1+.08*(snr-12.))/1000
        probability = 1/(1+math.exp(-(snr-8)/3.5))
        probability *= (1-.045*compression_level)
        probability = float(np.clip(probability,.02,.995))
        direct = a.source==a.receiver-1
        base_factor = (.43 if a.packet=='state' else .27)
        factor = base_factor+.115*compression_level+(.24 if not direct else 0.)
        factor = min(.94,factor)
        after = sigma.copy()
        r = a.receiver-1
        after[r] *= 1-probability*(1-factor)
        return after,probability,latency

    def rank(self,method):
        self.prepare()
        valid = [a for a in ACTIONS[1:] if a.bits_kb<=self.budget]
        if not valid:return [0]
        if method=='random':
            rng=np.random.default_rng(self.seed*1009+self.k*17)
            ordering=[int(x) for x in rng.permutation([a.id for a in valid])]
            chosen=[];spent=0.;receivers=set()
            for aid in ordering:
                a=ACTIONS[aid]
                if a.receiver in receivers or spent+a.bits_kb>self.budget:continue
                chosen.append(aid);receivers.add(a.receiver);spent+=a.bits_kb
            return chosen or [0]
        if method=='full_broadcast':
            # Budget-capped direct state broadcasts; one packet per receiver.
            chosen=[];spent=0.
            for receiver in np.argsort(-self.aoi)+1:
                options=[a for a in valid if a.receiver==receiver and
                         a.source==receiver-1 and a.packet=='state']
                options.sort(key=lambda a:a.bits_kb)
                if options and spent+options[0].bits_kb<=self.budget:
                    chosen.append(options[0].id);spent+=options[0].bits_kb
            return chosen or [0]
        chosen=[];spent=0.;receivers=set();current=self.sigma.copy()
        while True:
            available=[a for a in valid if a.receiver not in receivers and
                       spent+a.bits_kb<=self.budget]
            if not available:break
            before1=self.cbar(current,1)
            before10=self.cbar(current,10)
            ap_before=self.ap_proxy(current)
            scores=[]
            for a in available:
                after,p,delay=self.action_model(a,current)
                if method=='nearest':
                    score=-abs(a.receiver-a.source)+.002*p
                elif method=='ap':
                    score=(self.ap_proxy(after)-ap_before)-.12*delay
                elif method=='aoi':
                    score=self.aoi[a.receiver-1]*p/(a.bits_kb+.2)
                elif method=='state':
                    score=(np.linalg.norm(current)-np.linalg.norm(after))/(
                        a.bits_kb+.2)
                elif method=='cr_voi_1':
                    score=max(0,before1-self.cbar(after,1))/(
                        a.bits_kb+20*delay+.2)
                elif method=='cr_voi_10':
                    score=max(0,before10-self.cbar(after,10))/(
                        a.bits_kb+20*delay+.2)
                else:raise ValueError(method)
                scores.append((score,-a.bits_kb,-a.id,a.id,after))
            selected=max(scores,key=lambda x:x[:4])
            aid=selected[3];a=ACTIONS[aid]
            chosen.append(aid);receivers.add(a.receiver);spent+=a.bits_kb
            current=selected[4]
        return chosen or [0]

    def step(self,action_ids,method='policy'):
        self.prepare()
        if isinstance(action_ids,int):action_ids=[action_ids]
        if sum(ACTIONS[i].bits_kb for i in action_ids)>self.budget+1e-9:
            raise ValueError('Shared budget exceeded')
        k=self.k
        c_before=self.cbar(self.sigma)
        ap_before=self.ap_proxy(self.sigma)
        chosen=[];delivered=[];latencies=[];probabilities=[]
        bits=0.
        for aid in action_ids:
            a=ACTIONS[int(aid)]
            if a.id==0:continue
            bits+=a.bits_kb
            _,p,delay=self.action_model(a)
            success=self.trace['delivery'][k,a.id]<p
            chosen.append(a.id);latencies.append(delay);probabilities.append(p)
            if success:
                r=a.receiver-1
                direct=a.source==a.receiver-1
                factor=(.43 if a.packet=='state' else .27)+.115*(
                    COMPRESSIONS.index(a.compression))+(
                    .24 if not direct else 0.)
                factor=min(.94,factor)
                self.sigma[r]*=factor
                self.error[r]*=factor
                self.aoi[r]=delay
                delivered.append(a.id)
        c_after=self.cbar(self.sigma)
        c_true=self.ctrue(self.error)
        ap_after=self.ap_proxy(self.sigma)
        oldx=self.x.copy();oldv=self.v.copy();olda=self.a.copy()
        time=k*DT
        command=-3.6 if 2.0<=time<3.7 else 0.
        if self.scenario_name=='combined' and 4.0<=time<5.0:command=-2.0
        leader_tau=CONFIG['leader_tau_s']
        self.a[0]+= (1-math.exp(-DT/leader_tau))*(command-self.a[0])
        u_first=0.
        for i in range(1,4):
            gap=oldx[i-1]-oldx[i]
            e=gap-8-self.h*oldv[i]
            r=oldv[i-1]-oldv[i]
            estimate=self.error[i-1]
            u=(CONFIG['kp']*(e+estimate[0])+
               CONFIG['kd']*(r+estimate[1])+
               CONFIG['ka']*(olda[i-1]+estimate[2]))
            maxbrake=net_brake(self.scenario['load'],
                               GRADES[self.scenario['grade']])/self.params.b
            u=float(np.clip(u,-maxbrake,1.8))
            if i==1:u_first=u
            alpha=math.exp(-DT/self.params.tau)
            self.a[i]=alpha*olda[i]+(1-alpha)*self.params.b*u
        self.x += DT*oldv
        self.v += DT*olda
        gaps=self.x[:-1]-self.x[1:]
        spacing=gaps-(8+self.h*self.v[1:])
        physical_margin=float(np.min(gaps-10))
        max_abs_spacing=float(np.max(np.abs(spacing)))
        # AP proxy is generated by the same synthetic uncertainty model for
        # every method; it is not detector AP@IoU on TruckV2X.
        row=dict(scenario=self.scenario_name,seed=self.seed,budget_kb=self.budget,
            method=method,time_s=round((k+1)*DT,3),
            source='|'.join(str(ACTIONS[i].source) for i in chosen),
            receiver='|'.join(str(ACTIONS[i].receiver) for i in chosen),
            action_id='|'.join(map(str,chosen)) if chosen else '0',
            packet='|'.join(ACTIONS[i].packet for i in chosen),
            compression='|'.join(ACTIONS[i].compression for i in chosen),
            channel='|'.join(str(ACTIONS[i].channel) for i in chosen),
            bits_kb=bits,
            delivered_bits_kb=sum(ACTIONS[i].bits_kb for i in delivered),
            delivery_probability_mean=float(np.mean(probabilities)) if probabilities else 0.,
            latency_s=max(latencies,default=0.),
            delivered_count=len(delivered),aoi_s=float(np.mean(self.aoi)),
            snr_dB=float(np.mean(self.trace['snr'][k])),
            occluded_count=int(np.sum(self.trace['occ'][k])),
            c_before=c_before,c_after=c_after,c_true=c_true,
            delta_c=c_before-c_after,ap_proxy_before=ap_before,
            ap_proxy_after=ap_after,
            qualified=int(c_after<1),physical_margin_m=physical_margin,
            min_gap_m=float(np.min(gaps)),spacing_rmse_step_m=float(np.sqrt(
                np.mean(spacing**2))),peak_spacing_error_m=max_abs_spacing,
            brake_command_mps2=float(u_first),
            active_constraint_margin=1-c_after,
            fallback=int(len(delivered)==0))
        self.rows.append(row)
        self.k+=1
        self.ready=False
        done=self.k>=STEPS
        return row,done

def summarize(rows):
    return dict(scenario=rows[0]['scenario'],seed=rows[0]['seed'],
        method=rows[0]['method'],budget_kb=rows[0]['budget_kb'],
        mean_ap_proxy=float(np.mean([r['ap_proxy_after'] for r in rows])),
        delivered_kbit_per_s=float(sum(r['bits_kb'] for r in rows)/(STEPS*DT)),
        received_kbit_per_s=float(sum(r['delivered_bits_kb'] for r in rows)/(STEPS*DT)),
        mean_latency_ms=float(1000*np.mean([r['latency_s'] for r in rows])),
        mean_aoi_s=float(np.mean([r['aoi_s'] for r in rows])),
        mean_c=float(np.mean([r['c_after'] for r in rows])),
        qualified_fraction=float(np.mean([r['qualified'] for r in rows])),
        minimum_physical_margin_m=float(min(r['physical_margin_m'] for r in rows)),
        minimum_active_constraint_margin=float(min(r['active_constraint_margin']
                                                    for r in rows)),
        spacing_rmse_m=float(np.sqrt(np.mean(
            np.array([r['spacing_rmse_step_m'] for r in rows])**2))),
        peak_spacing_error_m=float(max(r['peak_spacing_error_m'] for r in rows)),
        fallback_fraction=float(np.mean([r['fallback'] for r in rows])))

def protocol():
    return dict(evidence='model-based simulator, not TruckV2X detector AP or field safety',
        actions=[a.__dict__ for a in ACTIONS],budgets_kb_per_slot=BUDGETS,
        dt_s=DT,steps=STEPS,seeds=SEEDS,methods=METHODS,
        shared_network_trace='same seed, scenario and time-indexed action delivery draws',
        shared_plant='v4 four-truck linear lag and fixed CACC gains',
        deterministic_allocator='All methods use the same per-slot budget and one-packet-per-receiver limit; scores are recomputed after each expected update',
        shared_error_generator='fit-town T2_A3 median shape with hypothetical fusion scale',
        AP='explicit smooth uncertainty-based AP proxy, not measured AP@IoU',
        shield='model-based Cbar<1 screen only; not empirical/field guarantee')

if __name__=='__main__':
    (OUT/'network_sim_protocol.json').write_text(json.dumps(protocol(),indent=2),
                                                  encoding='utf-8')
    env=NetworkPlatoonEnv('combined',10,101)
    while env.k<STEPS:
        env.step(env.rank('cr_voi_1'),'cr_voi_1')
    print(summarize(env.rows))
