# Manuscript-to-artifact map

| Manuscript evidence | Reference files | Reproduce |
|---|---|---|
| TableI, related-work positioning | `paper/main.tex`, `paper/references.bib` | Compile manuscript |
| TableII, compact objectives and IA/BPE | `results/joint/`, `results/supervision/` | `python3 tools/reproduce_results.py` → `original2000` |
| TableIII, feature and assembly references | `results/supplement/*_test_predictions.csv`, `results/supplement/asm_only/` | Same command → `feature_and_assembly_references` |
| TableIV, missing declaration context | `results/supplement/aug_runs/`, `results/supplement/speed_robustness.json` | Same command → `mixed_view`; historical per-seed aggregates |
| TableV, full and overlap-filtered predictions | `results/expanded/*_seed*/test_predictions.csv`, `dedup_masks.json` | Same command → `expanded` |
| TableVI, same-A100 latency/throughput/memory | `results/expanded/efficiency_summary.json` | Inspect measurements; rerun `tools/benchmark_inference.py` on your GPU |
| Original2,000-function pretrained comparison | `results/pretrained/*_seed*/` | Same command → `original2000` |
| All matched-seed gains and threshold checks | `results/expanded/full_filtered_per_seed.csv` | Same command → `generated/expanded_per_seed.csv` |
| Figure1, framework with C/assembly/IPC example | `paper/figures/framework.tex` and `.pdf` | Compile figure source with its existing TeX setup |
| Figure2, supervision prediction distributions | `results/supervision/`, `results/joint/` | `python tools/plot_ablation_predictions.py` |
| Tokenizer structure and coverage | `data/tokenizers/tokenizer_spec.json`, `provenance/*cache_manifest.json` | `tools/validate_tokenization.py`, `tools/build_caches.py` |
| Dataset and label fingerprints | `provenance/raw_data_manifest.json`, `source_cache_manifest.json` | `tools/verify_package.py --raw-data` |

`results/historical/pretrained_30min/` is not a source for current paper numbers. The archived optional external prediction figure shows the original2,000-function IPC-only pretrained comparison and is not included in the current manuscript.
