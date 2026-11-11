# Paper-to-artifact map

The table numbering refers to the bundled `paper/main.tex` snapshot.

| Paper content | Archived evidence | Reproduction |
|---|---|---|
| Positioning / related-work table | `paper/main.tex` | Compile manuscript; literature comparison, not an accuracy table |
| Supervision and IA/BPE comparison | `results/joint/`, `results/supervision/` | `tools/reproduce_results.py`, Table II |
| CodeBERT / UniXcoder comparison | `results/pretrained/` | Same script, Table III |
| Simple features and assembly-only reference | `results/supplement/*_test_predictions.csv`, `asm_only/` | Same script, Table IV |
| Missing declaration context / mixed-view training | `results/supplement/aug_runs/`, `speed_robustness.json` | Same script, Table V; stored per-seed metrics |
| External prediction scatter | Three-seed files in `results/joint/` and `results/pretrained/` | `tools/plot_predictions.py` |
| Supervision prediction scatter | Three-seed files in `results/joint/` and `results/supervision/` | `tools/plot_ablation_predictions.py` |
| Framework and concrete example | `paper/figures/framework.tex`, `provenance/figures/framework_example.json` | Bundled PDF; standalone TeX source |
| Cost and tokenization lengths | `results/supplement/speed_robustness.json`, cache manifests in `provenance/` | Original `speed_robustness.py` for new timings |
| Hardware correlations and filters | `results/supplement/hw_merged.jsonl`, measurements and source predictions | `tools/reproduce_hardware_summary.py` |

`generated/tables.json` retains unrounded metrics. R² is over log1p IPC, MAE over raw IPC. Plot scripts show all 2,000 functions for each of three seeds per model, rather than choosing a favorable seed. Model weights are optional for reconstructing these archived outputs.

The raw data splits are frozen historical JSONL, not newly sampled AnghaBench. `provenance/source_cache_manifest.json` preserves original fingerprints. `data/splits/` preserves exact probe/evaluation identities. `provenance/original_code/` retains pre-portability scripts; `provenance/portability_changes.json` identifies changes made for this artifact.
