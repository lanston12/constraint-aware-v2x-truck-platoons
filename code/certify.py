"""Finite-horizon platoon certificate under DECLARED stress assumptions.

The 48,560 kg gross combination mass comes from Li et al. (2019), Tables
3+4. All other dynamic/brake/road numbers in CONFIG are scenario assumptions
or values inherited from the user's local S3 script, not measured Xinjiang
fleet parameters. See results/certificates/parameter_ledger.csv.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import svdvals

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "certificates"
OUT.mkdir(parents=True, exist_ok=True)

CONFIG = dict(dt=0.1, horizon=80, n_trucks=4, speed_mps=20.0,
              standstill_gap_m=8.0, physical_gap_min_m=10.0,
              stopping_reserve_m=5.0, lateral_accel_cap_mps2=2.94,
              friction_lower=0.60, gravity=9.81,
              kp=0.55, kd=1.25, ka=0.15,
              leader_tau_s=0.55, terminal_error_halfwidth_m=8.0,
              brake_force_N=170000.0, brake_efficiency=0.78,
              base_age_s=0.10, model_gap_reserve_m=0.5,
              model_stop_reserve_m=1.0, model_speed_reserve_mps=0.2,
              model_brake_reserve_mps2=0.15)
LOADS = {
    "L1_light": dict(mass_kg=20000.0, tau_s=0.40, act_gain=1.00),
    "L2_medium": dict(mass_kg=35000.0, tau_s=0.55, act_gain=0.92),
    "L3_loaded": dict(mass_kg=48560.0, tau_s=0.70, act_gain=0.85),
}
GRADES = {"G1_level": 0.0, "G2_downhill": 3.0, "G3_severe": 6.5}
CURVES = {"K1_straight": 0.0, "K2_moderate": 0.005, "K3_sharp": 0.010}


@dataclass(frozen=True)
class Truck:
    mass: float
    tau: float
    b: float


def truck(load):
    p = LOADS[load]
    return Truck(p["mass_kg"], p["tau_s"], p["act_gain"])


def net_brake(load, grade_deg):
    p = LOADS[load]
    traction = CONFIG["friction_lower"] * CONFIG["gravity"] * math.cos(math.radians(grade_deg))
    mechanical = CONFIG["brake_efficiency"] * CONFIG["brake_force_N"] / p["mass_kg"]
    return min(traction, mechanical) - CONFIG["gravity"] * math.sin(math.radians(grade_deg))


def speed_limit(curvature):
    if curvature == 0:
        return 33.0  # scenario road limit, not a roll/jackknife certificate
    return min(33.0, math.sqrt(CONFIG["lateral_accel_cap_mps2"] / curvature))


def homogeneous_hinf(h, load):
    p = truck(load)
    kp, kd, ka = (CONFIG[k] for k in ("kp", "kd", "ka"))
    w = np.r_[0., np.logspace(-4, 2.5, 5000)]
    s = 1j * w
    c = kp + kd * s
    d = s*s * (p.tau*s + 1) + p.b * (1+h*s) * c
    g = p.b * (c + ka*s*s) / d
    me = -p.b * (1+h*s)*c/d
    ma = -p.b * (1+h*s)*ka/d
    poles = np.roots([p.tau, 1+p.b*h*kd, p.b*(h*kp+kd), p.b*kp])
    stable = bool(np.all(np.real(poles) < -1e-8))
    return float(np.max(np.abs(g))), float(np.max(np.sqrt(np.abs(me)**2+np.abs(ma)**2))), stable


def matrices(trucks, h):
    n, dt = len(trucks), CONFIG["dt"]
    F = np.eye(3*n)
    B0 = np.zeros(3*n)
    Berr = [np.zeros((3*n, 3)) for _ in range(n)]
    kp, kd, ka = (CONFIG[k] for k in ("kp", "kd", "ka"))
    for i, p in enumerate(trucks):
        j = 3*i
        F[j, j+1] = dt
        F[j+1, j+2] = dt
        alpha = math.exp(-dt/p.tau)
        F[j+2, :] = 0
        F[j+2, j+2] = alpha
        if i == 0:
            B0[j+2] = 1-alpha
        else:
            q = (1-alpha)*p.b
            F[j+2, 3*(i-1)] += q*kp
            F[j+2, j] -= q*kp
            F[j+2, 3*(i-1)+1] += q*kd
            F[j+2, j+1] -= q*(kd+kp*h)
            F[j+2, 3*(i-1)+2] += q*ka
            Berr[i][j+2, :] = q*np.array([kp, kd, ka])
    return F, B0, Berr


def outputs(n, h):
    ce, cv, cg, cp = [], [], [], []
    for i in range(1, n):
        e = np.zeros(3*n)
        e[3*(i-1)] = 1
        e[3*i] = -1
        e[3*i+1] = -h
        v = np.zeros(3*n); v[3*i+1] = 1
        gap = np.zeros(3*n); gap[3*(i-1)] = 1; gap[3*i] = -1
        pred = np.zeros(3*n); pred[3*(i-1)+1] = 1
        ce.append(e); cv.append(v); cg.append(gap); cp.append(pred)
    return ce, cv, cg, cp


def lifted(trucks, h, H=None):
    H = H or CONFIG["horizon"]
    n = len(trucks)
    F, B0, Berr = matrices(trucks, h)
    ce, cv, cg, cp = outputs(n, h)
    S0 = np.zeros((3*n, H))
    Se = [np.zeros((3*n, 3*H)) for _ in range(n)]
    E0 = [np.zeros((H, H)) for _ in range(n-1)]
    V0 = [np.zeros((H, H)) for _ in range(n)]
    # Every output must include ALL upstream/local perception channels; using
    # only channel i underestimates the downstream uncertainty gain.
    Eerr = [None] + [np.zeros((H, 3*H*(n-1))) for _ in range(1,n)]
    V_err = [None] + [np.zeros((H, 3*H*(n-1))) for _ in range(1,n)]
    G_err = [None] + [np.zeros((H, 3*H*(n-1))) for _ in range(1,n)]
    P_err = [None] + [np.zeros((H, 3*H*(n-1))) for _ in range(1,n)]
    U_err = [None] + [np.zeros((H, 3*H*(n-1))) for _ in range(1,n)]
    kp, kd, ka = (CONFIG[k] for k in ("kp", "kd", "ka"))
    for k in range(H):
        # Input at sample k propagates to state k+1; output sampled at k+1.
        S0 = F @ S0
        S0[:, k] += B0
        for i in range(1,n):
            Se[i] = F @ Se[i]
            Se[i][:, 3*k:3*k+3] += Berr[i]
        for i in range(n):
            V0[i][k, :] = S0[3*i+1, :]
        for i in range(1,n):
            E0[i-1][k, :] = ce[i-1] @ S0
            # u_i = kp*e_i + kd*(vprev-vi) + ka*aprev + direct error.
            cu = kp*ce[i-1] + kd*(cp[i-1]-cv[i-1])
            cu[3*(i-1)+2] += ka
            for j in range(1,i+1):
                sl=slice((j-1)*3*H,j*3*H)
                Eerr[i][k,sl] = ce[i-1] @ Se[j]
                V_err[i][k,sl] = cv[i-1] @ Se[j]
                G_err[i][k,sl] = cg[i-1] @ Se[j]
                P_err[i][k,sl] = cp[i-1] @ Se[j]
                U_err[i][k,sl] = cu @ Se[j]
            U_err[i][k, (i-1)*3*H+3*k:(i-1)*3*H+3*k+3] += [kp,kd,ka]
    return dict(E0=E0,V0=V0,Eerr=Eerr,Verr=V_err,Gerr=G_err,Perr=P_err,Uerr=U_err)


def factor_gain(Lprev, Lnext):
    # Exact finite-horizon e_next=H e_prev exists only if ker(Lprev) subset ker(Lnext).
    U, s, Vh = np.linalg.svd(Lprev, full_matrices=True)
    rank = int(np.sum(s > max(s[0]*1e-9, 1e-11))) if len(s) else 0
    if rank == 0:
        return math.inf, math.inf, rank
    leakage = np.linalg.norm(Lnext @ Vh[rank:].T, 2) if rank < len(s) else 0.0
    Hmap = (Lnext @ Vh[:rank].T / s[:rank]) @ U[:, :rank].T
    gain = float(svdvals(Hmap)[0])
    return gain, float(leakage), rank


def restricted_gain(Lprev, Lnext):
    """Exact gain on span of three declared leader-excitation profiles only."""
    H = Lprev.shape[1]
    t = np.arange(H)
    basis = np.column_stack([
        ((t >= H//8) & (t < H//4)).astype(float),
        np.sin(2*np.pi*t/H),
        np.convolve(np.random.default_rng(20260925).standard_normal(H),
                    np.ones(7)/7, mode="same"),
    ])
    basis /= np.linalg.norm(basis, axis=0)
    A, B = Lprev @ basis, Lnext @ basis
    s = np.linalg.svd(A, compute_uv=False)
    if s[-1] < 1e-9:
        return math.inf
    return float(svdvals(B @ np.linalg.pinv(A))[0])


def evaluate(load, grade_name, curve_name, h, age, hetero=False, H=None):
    H = H or CONFIG["horizon"]
    grade = GRADES[grade_name]
    curvature = CURVES[curve_name]
    if hetero:
        vehicles = [truck("L2_medium"),truck("L1_light"),truck("L2_medium"),truck("L3_loaded")]
    else:
        vehicles = [truck("L2_medium")] + [truck(load)]*3
    lin = lifted(vehicles, h, H)
    # Full-dimensional finite-horizon leader-to-spacing error factorization.
    links = []
    for idx in (1,2):
        links.append(factor_gain(lin["E0"][idx-1], lin["E0"][idx]))
    gamma = max(a[0] for a in links)
    leakage = max(a[1] for a in links)
    gamma_restricted = max(restricted_gain(lin["E0"][idx-1],lin["E0"][idx]) for idx in (1,2))
    beta = max(float(np.sqrt(np.linalg.eigvalsh(lin["Eerr"][i] @ lin["Eerr"][i].T)[-1])) for i in range(1,4))
    # Worst constraint gain across followers and time, 2-to-infinity.
    v = CONFIG["speed_mps"]
    rows = []
    for i in range(1,4):
        name = "L3_loaded" if hetero and i==3 else ("L2_medium" if hetero and i==2 else ("L1_light" if hetero else load))
        pred_name = "L2_medium" if i==1 else ("L1_light" if hetero and i==2 else ("L2_medium" if hetero and i==3 else load))
        bf = net_brake(name,grade)
        bp = net_brake(pred_name,grade)
        gap = CONFIG["standstill_gap_m"] + h*v
        margins = {
            "spacing": gap-CONFIG["physical_gap_min_m"]-CONFIG["model_gap_reserve_m"],
            "stopping": gap-CONFIG["stopping_reserve_m"]-v*(age+vehicles[i].tau)
                        -v*v/(2*bf)+v*v/(2*bp)-CONFIG["model_stop_reserve_m"] if bf>0 and bp>0 else -math.inf,
            "speed_envelope": speed_limit(curvature)-v-CONFIG["model_speed_reserve_mps"],
            "brake_input": bf/vehicles[i].b-CONFIG["model_brake_reserve_mps2"],
            "terminal": CONFIG["terminal_error_halfwidth_m"]-CONFIG["model_gap_reserve_m"],
        }
        stop_map = (lin["Gerr"][i] - (age+vehicles[i].tau+v/bf)*lin["Verr"][i]
                    + (v/bp)*lin["Perr"][i]) if bf>0 and bp>0 else np.zeros_like(lin["Gerr"][i])
        maps = {"spacing": lin["Gerr"][i], "stopping": stop_map,
                "speed_envelope": lin["Verr"][i], "brake_input": lin["Uerr"][i],
                "terminal": lin["Eerr"][i][-1:, :]}
        for constraint, margin in margins.items():
            eta = float(np.max(np.linalg.norm(maps[constraint], axis=1)))
            # eps bounds each of the three follower error-vector sample norms.
            eps = margin / (eta * math.sqrt(H*(len(vehicles)-1))) if eta > 0 else math.inf
            rows.append((eps, constraint, margin, eta, i, name))
    active = min(rows, key=lambda x:x[0])
    gamma_hinf, beta_hinf, stable = homogeneous_hinf(h, load)
    reason = "certified_finite_horizon"
    feasible = True
    if not stable:
        reason, feasible = "unstable_homogeneous_model", False
    elif leakage > 1e-6:
        reason, feasible = "error_factorization_failed", False
    elif gamma > 1.0+1e-6:
        reason, feasible = "gamma_gt_one", False
    elif active[2] <= 0:
        reason, feasible = "nonpositive_" + active[1] + "_slack", False
    epscrit = float(active[0]) if feasible else float("nan")
    restricted_feasible = bool(stable and gamma_restricted <= 1.0+1e-6 and active[2] > 0)
    return dict(load=load if not hetero else "heterogeneous_L2_L1_L2_L3",
                grade=grade_name, grade_deg=grade, curvature=curve_name,
                curvature_inv_m=curvature, headway_s=h, age_s=age,
                tau_s=vehicles[-1].tau, gamma=gamma, gamma_restricted=gamma_restricted,
                gamma_hinf_homogeneous=gamma_hinf,
                beta=beta, beta_hinf_homogeneous=beta_hinf,
                eta_active=active[3], active_constraint=active[1],
                active_follower=active[4], margin=active[2],
                epsilon_crit=epscrit, feasible=int(feasible),
                epsilon_crit_restricted=float(active[0]) if restricted_feasible else float("nan"),
                feasible_restricted=int(restricted_feasible),
                certificate_reason=reason, factorization_leakage=leakage,
                horizon_steps=H, horizon_s=H*CONFIG["dt"],
                max_gain_rank=min(x[2] for x in links))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    with (OUT / "config.json").open("w",encoding="utf-8") as f:
        json.dump(dict(CONFIG=CONFIG,LOADS=LOADS,GRADES=GRADES,CURVES=CURVES),f,indent=2)
    rows=[]
    for load in LOADS:
        for grade in GRADES:
            for curve in CURVES:
                for h in [1.0,1.5,2.0,2.5,3.0,4.0]:
                    for age in [0.0,0.2,0.4]:
                        rows.append(evaluate(load,grade,curve,h,age))
        print("completed",load,len(rows),flush=True)
    for h in [1.5,2.0,3.0,4.0]:
        rows.append(evaluate("L3_loaded","G2_downhill","K2_moderate",h,0.2,hetero=True))
    write_csv(OUT / "certificate_table.csv", rows)
    print("rows",len(rows),"feasible",sum(x["feasible"] for x in rows),flush=True)
    print("reasons",{x["certificate_reason"]:sum(y["certificate_reason"]==x["certificate_reason"] for y in rows) for x in rows},flush=True)


if __name__ == "__main__":
    main()
