# Submission figure evidence matrix

Fig.1 uses the author-supplied PNG unchanged; all numerical result figures retain PDF/PNG assets and unchanged raw sources. Internal data-directory names are provenance only and do not appear in submitted figures. The architecture is a schematic, so it has no fabricated raw-data table.

| Figure | Content | Evidence type | Raw source | Numeric support | Boundary |
|---|---|---|---|---|---|
| 1 | Architecture | Schematic; no numerical sample | figures/fig1.png; author-supplied main-folder fig1.png; saved model/protocol descriptions | Perception error → consumption → CRIH → value | Diagram is an interface, not field evidence |
| 2 | Empirical relevance | Empirical one-update + exact model rows | results/calibration/fresh_certificate_comparison.csv; fresh_comparison_metrics.csv; fresh_diagnostics.csv; results/certificates/directional_axes_517.csv | 212/223; median1.149;84passes/1false;rho.819 over325;262/517 | No verified predecessor identity or complete empiricalH80 stack |
| 3 | CRIH | Generated complete temporal sequences | results/v5/horizon_metrics.csv; horizon_trajectories.csv; horizon_active_constraints.csv | H80 nominal.971/loaded.033/combined0;combinedmedian4.519 | Below-.95 coverage retained; assumed fusion; CRIH is descriptive |
| 4 | Actual received-bit Pareto | Paired model-based closed loop | results/v6/received_pareto_points.csv; received_pareto_cell_summary.csv; received_pareto_frontier.csv | Cap10 CR.389/68.34 versusAP.477/67.75;State.353/69.59 | Received rates policy-dependent; no equal-rate causal or global-dominance claim |
| 5 | Budget--stress map | 1728 model-based episodes | results/v6/regime_episode_metrics.csv; regime_contrasts.csv; regime_active_best.csv; regime_protocol.json | AP24/36;AoI36/36;State0/36 CRwins;bestState30/AP6 | Brake-modal all36; CR not best anywhere |
| 6 | Combined mechanism | Fixed seed201/cap10, common-state candidate scores | results/v6/mechanism_trace.csv; mechanism_candidates.csv; mechanism_selected_events.csv; mechanism_all_event_scores.csv; mechanism_protocol.json | Automatic5.0/5.7s;5.7CR3kbitreceiver2 expectedDeltaC.167 | Expected single packet differs from realized whole slot |
| S1 | AP-control disagreement | 432 paired model episodes | results/v6/ap_control_disagreement.csv; ap_control_disagreement_summary.csv; ap_control_representative.csv | 156/43236.1%;fixedmax-product example | AP proxy only; no cherry-picking |
| S2 | PPO/shield | Model-based inspired policy | results/v5/gsppo_heldout_metrics.csv; gsppo_paired_summary.csv; gsppo_training_history.csv | CRminusFan-.1355;shieldminusCR+.1386 | Surrogate reproduction scope; shield negative |
| S3 | Diversity versus gain | 576 pairs /72 neutral references | results/v7/fig7_diversity_vs_advantage_raw.csv; fig7_diversity_win_rates.csv; reference_diversity.csv | Highentropy49/304VoIwins;96/304raw wins | No consistent switching-specific gain |
| S4 | Diversity--budget map | Descriptive entropy groups | results/v7/fig8_constraint_regime_map_raw.csv; fig8_best_method_raw.csv; paired_contrasts.csv | R6 gaps+.0128/+.0126 | R1nonpositive margins; bins not causal effects |
| S5 | Cost divisor | 576 paired episodes/common-state action ranks | results/v7/fig9_cr_denominator_episodes_raw.csv; fig9_ranking_disagreement_raw.csv; denominator_common_state_ranking.csv | .2176/89.26 versus.2257/60.99;rankdifferencesretained | Higher rate prevents equal-bit superiority claim |
| S6 | Switching trace | Automatic neutral-reference selection | results/v7/fig10_switching_trace_raw.csv; fig10_switching_reference_raw.csv; fig10_switching_selection.csv | R4seed212nine reference switches | Scheduled policies remain brake-modal |

## Final v9 interpretation audit

All numerical raw sources are unchanged from v8. Fig.6 was re-rendered with its geographic title replaced; its data and plotted quantities are unchanged. Fig.1 now uses the author-supplied PNG. Final figure placement and every page were visually inspected.

- Fig.2: declared linear row/probe geometry and held-out one-update errors, not exact saturated command consumption.
- Fig.4: the scheduling objective is also a diagnostic; actual received rates remain policy-dependent; State and independent physical metrics prevent a universal/physical-benefit reading.
- Fig.5: unchanged core boundary: CR below AP24/36, below AoI36/36, below State0/36; best State30/AP6/CR0; brake modal36/36.
- Fig.6: common pre-decision state, expected candidate restoration, explicit brake-probe timing; applied command and fallback are separately defined. It illustrates the mechanism, not superior physical performance.
- Supplement S2–S6: all negative State, diversity, denominator, shield and physical-response findings retained. No new graph or numerical sample was introduced.

## Author-supplied Fig.1 provenance

Source PNG SHA-256: `cdfa85647dadefe915ab87ae878c60e099cfe50c7c355978bcee442a5d40d220`. The copied image is byte-identical to the supplied source. It is an architecture schematic, not a new numerical result. Its caption was updated to describe the displayed stages; no formula-label correction note is included, as requested by the author.

## Neutral heavy-duty freight terminology

The manuscript and submitted figures use combined heavy-duty stress and combined-stress mechanism study. The Introduction contains one general freight-corridor motivation sentence naming load, road-grade, curvature, occlusion and communication stress. Geographic and coal-specific framing is absent from the compiled text. Fig.6 PDF/PNG was regenerated only from frozen inputs; numerical results and the35-reference selection are unchanged. Updated Introduction, scope, mechanism figure and subsection pages were rendered and visually inspected.

