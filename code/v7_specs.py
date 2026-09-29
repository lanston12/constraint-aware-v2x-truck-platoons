"""Predeclared physical and paired scenario grid; no outcome-based selection."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/v7'
OUT.mkdir(exist_ok=True)
BUDGETS=(5.,6.,8.,10.,12.,16.,20.,24.)
SEEDS=tuple(range(201,213))
METHODS=('ap','aoi','state','cr_delta_c','cr_voi')
SCENARIOS=(
    dict(name='R1_brake',load='L3_loaded',speed0=20.,headway=3.,
         age=.2,gap_offset=0.,closing_offset=0.,uncertainty_factor=1.5,
         network_stress=True,phases=[
             dict(start=0,end=2,grade='G2_downhill',curve='K1_straight',lead_command=0.),
             dict(start=2,end=4,grade='G2_downhill',curve='K1_straight',lead_command=-3.6),
             dict(start=4,end=8,grade='G2_downhill',curve='K1_straight',lead_command=0.)]),
    dict(name='R2_spacing',load='L1_light',speed0=20.,headway=1.,
         age=.2,gap_offset=-10.,closing_offset=.8,uncertainty_factor=1.05,
         network_stress=False,phases=[
             dict(start=0,end=2,grade='G1_level',curve='K1_straight',lead_command=0.),
             dict(start=2,end=3,grade='G1_level',curve='K1_straight',lead_command=-1.5),
             dict(start=3,end=8,grade='G1_level',curve='K1_straight',lead_command=0.)]),
    dict(name='R3_stopping',load='L1_light',speed0=20.,headway=1.,
         age=.4,gap_offset=-2.,closing_offset=1.5,uncertainty_factor=1.12,
         network_stress=False,phases=[
             dict(start=0,end=2,grade='G1_level',curve='K1_straight',lead_command=0.),
             dict(start=2,end=3.5,grade='G1_level',curve='K1_straight',lead_command=-2.),
             dict(start=3.5,end=8,grade='G1_level',curve='K1_straight',lead_command=0.)]),
    dict(name='R4_speed',load='L1_light',speed0=23.5,headway=2.,
         age=.2,gap_offset=0.,closing_offset=0.,uncertainty_factor=1.08,
         network_stress=False,phases=[
             dict(start=0,end=3,grade='G1_level',curve='K2_moderate',lead_command=0.),
             dict(start=3,end=4,grade='G1_level',curve='K2_moderate',lead_command=-.8),
             dict(start=4,end=8,grade='G1_level',curve='K2_moderate',lead_command=0.)]),
    dict(name='R5_mixed',load='L2_medium',speed0=20.,headway=2.,
         age=.2,gap_offset=-2.,closing_offset=0.,uncertainty_factor=1.25,
         network_stress=True,phases=[
             dict(start=0,end=2,grade='G1_level',curve='K1_straight',lead_command=0.),
             dict(start=2,end=4,grade='G2_downhill',curve='K1_straight',lead_command=0.),
             dict(start=4,end=6,grade='G2_downhill',curve='K2_moderate',lead_command=0.),
             dict(start=6,end=8,grade='G2_downhill',curve='K2_moderate',lead_command=-2.5)]),
    dict(name='R6_switch',load='L1_light',speed0=23.,headway=1.,
         age=.4,gap_offset=-6.,closing_offset=.6,uncertainty_factor=1.35,
         network_stress=True,phases=[
             dict(start=0,end=2,grade='G1_level',curve='K1_straight',lead_command=0.),
             dict(start=2,end=4,grade='G1_level',curve='K2_moderate',lead_command=0.),
             dict(start=4,end=6,grade='G2_downhill',curve='K2_moderate',lead_command=-2.),
             dict(start=6,end=8,grade='G2_downhill',curve='K1_straight',lead_command=-1.)]),
)
BY_NAME={s['name']:s for s in SCENARIOS}

def write_protocol():
    protocol=dict(evidence='model-based physical stress extension',
                  source='frozen v4 row maps and v6 action/plant/network model',
                  scenarios=SCENARIOS,budgets_kbit_per_slot=BUDGETS,
                  paired_seeds=SEEDS,methods=METHODS,steps=80,dt_s=.1,
                  horizon_row_choice='one existing future row per follower/family, max frozen step-zero sensitivity / positive static margin; chosen before episodes',
                  dynamic_margin='current true gap/speed and phase; frozen leader/disturbance reserve; positive-floor .25 solely for diagnostic computation; nonpositive slots flagged',
                  diversity_reference='no-packet policy-neutral episode per scenario/seed',
                  denominator='bits(kbit)+20*latency(s)+.2, inherited v6',
                  QP='no QP layer in frozen controller; count recorded as zero')
    (OUT/'scenario_protocol.json').write_text(
        json.dumps(protocol,indent=2),encoding='utf-8')
    return protocol

if __name__=='__main__':
    write_protocol()
    print(OUT/'scenario_protocol.json')
