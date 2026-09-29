# Minimal reproduction package

**From Perception Error to Communication Value: Constraint-Aware V2X Information Scheduling for Heavy-Duty Truck Platoons**

## Quick start

Tested with CPython 3.9 on Windows and the exact package versions in `requirements.txt`.
Create an environment, install dependencies, then run from this directory:

```shell
python -m pip install -r requirements.txt
python reproduce.py
```

The pinned environment is intended for Python 3.9. If using a newer Python, create a Python 3.9 environment first. The script uses paths relative to its own location and requires no network after dependency installation. Expected runtime is a few minutes on a desktop computer.

## What this reproduces

- Main-paper Figs. 2–6 and all six supplementary result figures, as PDF and PNG in `figures/`.
- Fig. 1 is the supplied conceptual illustration and is included unchanged as `figures/fig1.png`.
- Main 1,728-episode paired sweep, the 432 AP–CR pairs, all received-bit Pareto point means, paired t intervals, disagreement counts and deterministic maximum-disagreement selection, recomputed from episode-level CSVs.
- The empirical 212/223 directional-bound coverage and error/consumption correlation, recomputed from the saved held-out matched observations.
- An independent assembly of the common-input linear maps for five existing load schedules.
- Four model episodes: combined stress, 10 kbit/slot, seed 201, with AP, AoI, State and CR. The original simulator reruns these episodes and compares every saved summary metric with the frozen study. This is the seed/condition used in main Fig. 6.

`audit/` contains regenerated statistics, the replayed slot traces, matrix consistency results, and `reproduction_status.json`. The script validates SHA-256 checksums before and after execution and blocks writes to `results/`.

## Data and evidence scope

The `results/` directory contains the minimum complete input tables used by the included plotting functions, saved model row maps, paired episode summaries, and all supplementary slot metrics needed to support automatic trace selection. Retaining the complete supplementary slot table avoids manual trace selection; no exploratory or rejected experiments are included.

`results/calibration/` contains processed held-out TruckV2X matching/error records, not raw point clouds or a perception-backbone training dataset. `results/perception/fit_town_tracking_errors.csv` supplies the original simulator's fitted scale; it is separate from the held-out empirical records. `results/v5/` contains generated temporal-horizon and saved PPO evidence. `results/v6/` contains the unified paired sweep and mechanism decisions. `results/v7/` contains the constraint-diversity extension. Protocol JSON files record assumptions, budgets and seeds. Internal version identifiers are retained for data provenance.

This compact package reproduces the reported result figures and selected deterministic simulations from frozen evidence. It does not rerun raw TruckV2X preprocessing, train perception networks, retrain PPO, or regenerate every research campaign. PPO figures use the saved training/held-out logs; the shielded PPO remains a negative ablation. Model AP proxy is not detector AP; model constraint consumption is not physical safety certification. See the paper's Experimental Scope/Limitations and `figure_evidence_matrix.md`.

## Files and provenance

- `code/`: original model, scheduler, lift, plotting and protocol implementations used in the final manuscript.
- `frozen_inputs_manifest.json`: relative source/data paths, sizes and SHA-256 checksums; copied input files are byte-identical to the final project.
- `references.bib`: the manuscript's 35-reference bibliography, including the TruckV2X source.
- `figure_evidence_matrix.md`: figure-to-evidence mapping carried over from the final project; historical paths describe the original research workspace, and only the dependencies listed above are included here.

Use `reproduce.py` as the supported entry point. Some original research scripts have full-campaign entry points that write results and are retained to show the implementation; the minimal workflow deliberately uses their functions with frozen inputs instead.

## Frozen submission release

Repository: https://github.com/lanston12/constraint-aware-v2x-truck-platoons

The original, previously tested minimal package is attached to release `v1.0.0`:
https://github.com/lanston12/constraint-aware-v2x-truck-platoons/releases/tag/v1.0.0

Direct ZIP download:
https://github.com/lanston12/constraint-aware-v2x-truck-platoons/releases/download/v1.0.0/TVT_minimal_reproduction.zip

ZIP SHA-256: `8038196eb7410afa4625cf84464a495bc51252c0bc20896575cfc11042a77644`.

The repository files and release archive contain the same frozen scientific inputs. The repository adds this release description and Git housekeeping files; the original ZIP remains unchanged. Expected outputs from the prior independent verification are included under `expected/`.
