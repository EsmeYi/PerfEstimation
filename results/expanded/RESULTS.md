# Completed supplementary evaluation — 2026-11-10

All21 frozen-checkpoint evaluations completed (7conditions ×3seeds). No optimizer updates or test-based selection. Every expanded run exactly reproduced its original2000 predictions (maximum absolute transformed-prediction difference0). Full test n=17,301; primary train-overlap-filtered subset n=12,600.

Filter: remove test functions whose whole-source alpha-normalized C lexical token5-gram set Jaccard is >=0.9 against any of309,913 training functions. Additional0.8/0.95 threshold results are in full_filtered_summary.json. This is a sensitivity analysis, not a project-disjoint split.

## Transformed-target R² (mean ± sample SD over3seeds)

| Condition | Full17,301 | Filtered12,600 |
|---|---:|---:|
| compact_ipc | 0.593163 ± 0.008504 | 0.493416 ± 0.007536 |
| compact_bpe_joint | 0.747750 ± 0.002698 | 0.645315 ± 0.004168 |
| compact_ia_joint | 0.758773 ± 0.000367 | 0.660542 ± 0.001904 |
| codebert_ipc | 0.889454 ± 0.003149 | 0.827582 ± 0.004523 |
| codebert_joint | 0.899881 ± 0.002262 | 0.843428 ± 0.003084 |
| unixcoder_ipc | 0.909114 ± 0.002969 | 0.857586 ± 0.004762 |
| unixcoder_joint | 0.913718 ± 0.000766 | 0.865610 ± 0.001268 |

## Paired joint-minus-IPC R² changes

| Encoder / scope | Seed42 | Seed43 | Seed44 |
|---|---:|---:|---:|
| compact / all | +0.156256 | +0.167063 | +0.173511 |
| compact / near09 | +0.159823 | +0.171200 | +0.170352 |
| codebert / all | +0.009290 | +0.014417 | +0.007573 |
| codebert / near09 | +0.014219 | +0.021007 | +0.012311 |
| unixcoder / all | +0.005020 | +0.008040 | +0.000754 |
| unixcoder / near09 | +0.009318 | +0.013090 | +0.001663 |

All three encoders improve in R² at every matched seed on both expanded scopes. The original2000 UniXcoder result remains mixed across seeds; retain the distinction between evaluation scopes.

## Same-A100 inference efficiency

FP32,512-token padded inputs; source encoding+scalar linear readout only, excluding host tokenization and transfers. Same physical GPU,100 repeats following20warmups. Historical pretrained weights used solely for architecture timing, never for new predictive comparisons.

| Encoder | Mean batch1 ms | Batch16 functions/s | Batch1 peak allocated MiB |
|---|---:|---:|---:|
| compact | 1.787 | 2250.4 | 48.7 |
| codebert | 8.983 | 167.6 | 500.4 |
| unixcoder | 8.919 | 167.7 | 505.2 |

Spearman and IPC MAE are available for every condition and filter in full_filtered_summary.json and full_filtered_per_seed.csv. Original2000 metrics are preserved separately.
Predictive labels remain static llvm-mca IPC, not measured hardware runtime.
Paper has not yet been updated with these supplementary results.
