"""Trajectory-level *model-based* certificate consumption for H=1..80.

The row maps are exact frozen v4 control geometry; error sequences are newly
generated synthetic cooperative-fusion traces. No TruckV2X 80-step label is
implied. Each full trajectory, rather than an individual frame, is one
calibration/test unit.
"""
import csv
import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import beta
from directional_certificate import compute

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/v5'
OUT.mkdir(parents=True, exist_ok=True)
HORIZONS = (1, 5, 10, 20, 40, 80)
SCENARIOS = (
    dict(name='nominal', kind='homogeneous', load='L1_light', grade='G1_level',
         curvature='K1_straight', headway_s=2., age_s=.2, uncertainty_factor=1.),
    dict(name='loaded', kind='homogeneous', load='L3_loaded', grade='G1_level',
         curvature='K1_straight', headway_s=3., age_s=.2, uncertainty_factor=1.12),
    dict(name='downhill', kind='homogeneous', load='L1_light', grade='G2_downhill',
         curvature='K1_straight', headway_s=2., age_s=.2, uncertainty_factor=1.08),
    dict(name='curve', kind='homogeneous', load='L1_light', grade='G1_level',
         curvature='K2_moderate', headway_s=2., age_s=.2, uncertainty_factor=1.10),
    dict(name='high_aoi', kind='homogeneous', load='L1_light', grade='G1_level',
         curvature='K1_straight', headway_s=2., age_s=.4, uncertainty_factor=1.35),
    dict(name='combined', kind='homogeneous', load='L3_loaded', grade='G2_downhill',
         curvature='K2_moderate', headway_s=3., age_s=.4, uncertainty_factor=1.75),
)

def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def fit_reference_scale():
    """Fit-town T2_A3 observed medians; fusion gains are explicit assumptions."""
    path = ROOT / 'results/perception/fit_town_tracking_errors.csv'
    with path.open(newline='', encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f)
                if r['tracker'] == 'T2_A3' and r['mode'] == 'observed']
    med = np.median(np.abs(np.array([
        [float(r[k]) for k in ('error_gap_m',
                              'error_relative_velocity_mps',
                              'error_acceleration_mps2')] for r in rows])), axis=0)
    gains = np.array([.25, .12, 1.0])
    std = med / .67448975 * gains
    return med, gains, std, len(rows)

def condition_map(s):
    folder = OUT / 'row_maps'
    folder.mkdir(exist_ok=True)
    file = folder / (s['name'] + '.npz')
    if file.exists():
        d = np.load(file)
        return d['F'], d['M'], d['step'], d['constraint']
    desc = {k: s[k] for k in ('kind', 'load', 'grade', 'curvature',
                              'headway_s', 'age_s')}
    row, F, meta = compute(desc, True)
    if not row['feasible']:
        raise ValueError('Nonpositive frozen margin: ' + s['name'])
    M = np.array([m['margin'] for m in meta])
    step = np.array([m['step'] for m in meta])
    constraint = np.array([m['constraint'] for m in meta])
    np.savez_compressed(file, F=F, M=M, step=step, constraint=constraint)
    return F, M, step, constraint

def sequence_batch(s, n, seed, base_std):
    rng = np.random.default_rng(seed)
    quality = rng.uniform(.50, 1.50, n)
    snr = rng.normal(16., 3.5, (n, 80)).clip(4., 28.)
    compression = rng.choice([0., .25, .5], size=(n, 80),
                             p=[.30, .45, .25])
    occlusion = rng.random((n, 80)) < .08
    aoi = s['age_s'] + rng.exponential(.12, (n, 80))
    factor = (1. + .025*np.maximum(20.-snr, 0.)) * (
        1. + .32*compression) * (1. + .60*occlusion) * (
        1. + .35*aoi)
    # Decision-time exposure uses ONLY the current quality estimate, SNR,
    # compression, occlusion and age. Future network realizations are hidden.
    exposure = quality * s['uncertainty_factor'] * factor[:,0]
    innovation = rng.standard_t(6, size=(n, 80, 3, 3)) / math.sqrt(1.5)
    error = np.zeros_like(innovation)
    phi = .75
    for t in range(80):
        previous = error[:, t-1] if t else 0.
        error[:, t] = phi*previous + math.sqrt(1-phi*phi)*innovation[:, t]
    error *= (quality*s['uncertainty_factor'])[:, None, None, None]
    error *= factor[:, :, None, None]
    error *= base_std[None, None, None, :]
    return error, exposure, quality, snr, compression, occlusion, aoi

def finite_quantile(x, alpha=.05):
    a = np.sort(np.asarray(x, float))
    rank = math.ceil((len(a)+1)*(1-alpha))
    return float(a[rank-1]) if rank <= len(a) else float('inf')

def proportion_interval(success, n):
    # Two-sided exact binomial interval; a descriptive reference for
    # independent generated trajectories, not a field coverage assertion.
    low = 0. if success == 0 else float(beta.ppf(.025, success, n-success+1))
    high = 1. if success == n else float(beta.ppf(.975, success+1, n-success))
    return low, high

def main():
    med, gains, std, fit_n = fit_reference_scale()
    protocol = {
        'evidence': 'model-based synthetic error sequences; not empirical 80-step TruckV2X',
        'seed': 20260925, 'calibration_sequences_per_scenario': 160,
        'test_sequences_per_scenario': 240, 'horizons': HORIZONS,
        'fit_town_observed_T2_A3_matched_n': fit_n,
        'fit_town_median_abs_error': med.tolist(),
        'hypothetical_cooperative_fusion_gain': gains.tolist(),
        'base_innovation_std': std.tolist(),
        'temporal_AR_coefficient': .75, 'innovation': 'variance-normalized Student-t df=6',
        'snr_dB': 'Normal(16, 3.5) clipped to [4,28]',
        'compression': [0, .25, .5], 'occlusion_probability': .08,
        'AoI': 'condition age + exponential(mean .12 s)',
        'predictor_information': 'decision-time quality scale and step-zero network/perception covariates only; no future exposure',
        'calibration_unit': 'one complete independently seeded 80-step trajectory',
        'scenarios': SCENARIOS,
        'row_source': 'frozen v4 common-input maps; every listed constraint row'
    }
    (OUT/'horizon_protocol.json').write_text(json.dumps(protocol, indent=2),
                                             encoding='utf-8')
    detail = []
    summary = []
    active = []
    for idx, s in enumerate(SCENARIOS):
        F, M, step, constraint = condition_map(s)
        err, exposure, quality, snr, comp, occ, aoi = sequence_batch(
            s, 400, 20260925 + idx*101, std)
        # The frozen lift orders follower, then time, then channel.
        X = err.transpose(0, 2, 1, 3).reshape(400, 720)
        normalized = F/M[:, None]
        projected = np.abs(normalized @ X.T)
        truth = np.stack([np.max(projected[step < H], axis=0)
                          for H in HORIZONS], axis=1)
        cal = slice(0, 160)
        test = slice(160, None)
        support = np.quantile(exposure[cal], [.01, .99])
        accepted = (exposure[test] >= support[0]) & (
            exposure[test] <= support[1])
        bounds = np.zeros((240, len(HORIZONS)))
        for hi, H in enumerate(HORIZONS):
            q = finite_quantile(truth[cal, hi]/exposure[cal])
            bounds[:, hi] = q*exposure[test]
            t = truth[test, hi]
            passmask = (bounds[:, hi] < 1) & accepted
            covered = t <= bounds[:, hi]
            sel = accepted
            false = passmask & (t > 1)
            lo, upper = proportion_interval(int(np.sum(covered & sel)),
                                            int(np.sum(sel)))
            summary.append(dict(scenario=s['name'], H=H,
                n_test=240, n_accepted=int(sel.sum()),
                coverage=float(np.mean(covered[sel])),
                coverage_ci_low=lo, coverage_ci_high=upper,
                median_Cbar=float(np.median(bounds[sel, hi])),
                qualified_fraction=float(np.mean(passmask)),
                qualified_fraction_accepted=float(np.mean(passmask[sel])),
                false_pass_count=int(false.sum()),
                false_pass_per_pass=float(false.sum()/passmask.sum())
                    if passmask.any() else float('nan'),
                abstention_rate=float(1-np.mean(sel)),
                median_Ctrue=float(np.median(t[sel])),
                calibration_quantile=q))
            arg = np.argmax(projected[step < H, 160:], axis=0)
            winning = constraint[step < H][arg]
            for family in ('spacing','stopping','speed_envelope','brake_input','terminal'):
                active.append(dict(scenario=s['name'], H=H,
                    constraint=family, fraction=float(np.mean(winning==family))))
            for j in range(240):
                detail.append(dict(scenario=s['name'], trajectory=j, H=H,
                    quality=float(quality[160+j]), mean_snr_dB=float(np.mean(snr[160+j])),
                    mean_compression=float(np.mean(comp[160+j])),
                    occlusion_fraction=float(np.mean(occ[160+j])),
                    mean_aoi_s=float(np.mean(aoi[160+j])),
                    C_true=float(t[j]), C_bar=float(bounds[j,hi]),
                    accepted=int(accepted[j]), qualified=int(passmask[j]),
                    active_constraint=str(winning[j])))
        hcert = np.max(np.where((bounds < 1) & accepted[:,None],
                                np.array(HORIZONS)[None,:], 0), axis=1)
        print(s['name'], 'Hcert median', np.median(hcert),
              'H80 pass', summary[-1]['qualified_fraction'], flush=True)
    write_csv(OUT/'horizon_trajectories.csv', detail)
    write_csv(OUT/'horizon_metrics.csv', summary)
    write_csv(OUT/'horizon_active_constraints.csv', active)
    print('wrote', len(detail), 'trajectory-horizon rows', flush=True)

if __name__ == '__main__':
    main()
