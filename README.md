# Source-Level Static Performance Estimation through Assembly-Supervised Learning

Replication package for the manuscript in [`paper/`](paper/). Paired assembly and static IPC labels supervise source-code representations during training. Prediction uses source code alone. The package covers a compact Transformer, CodeBERT, and UniXcoder, including matched IPC-only and joint-supervision conditions.

## Reproduce the results without a GPU

Python 3.11 or newer is sufficient. No third-party packages, model downloads, credentials, or cluster access are required:

```bash
git clone https://github.com/EsmeYi/PerfEstimation.git
cd PerfEstimation
python3 tools/verify_package.py
python3 tools/reproduce_results.py
```

The second command recomputes R², IPC MAE, and Spearman correlation from **53 prediction files**, verifies common function identities and labels, and checks the expanded results against their archived summaries. Outputs are `generated/tables.md`, `generated/tables.json`, and `generated/expanded_per_seed.csv`.

Final R², mean ± sample SD across seeds 42, 43, and 44:

| Source encoder / supervision | All 17,301 test functions | Filtered 12,600 functions |
|---|---:|---:|
| CodeBERT / IPC | .8895 ± .0031 | .8276 ± .0045 |
| CodeBERT / joint | .8999 ± .0023 | .8434 ± .0031 |
| UniXcoder / IPC | .9091 ± .0030 | .8576 ± .0048 |
| UniXcoder / joint | .9137 ± .0008 | .8656 ± .0013 |
| Compact / IPC | .5932 ± .0085 | .4934 ± .0075 |
| Compact / joint, BPE assembly | .7478 ± .0027 | .6453 ± .0042 |
| Compact / joint, IA assembly | .7588 ± .0004 | .6605 ± .0019 |

R² uses `log1p(IPC)`. Filtering removes test functions with normalized lexical token 5-gram Jaccard similarity ≥0.9 to any encoder-training function. Scores for the original 2,000-function controlled sample and sensitivity thresholds 0.8/0.95 are retained separately.

## Evaluate a frozen model

Large artifacts are distributed through the [Release](https://github.com/EsmeYi/PerfEstimation/releases/tag/v2026.11.10), with download sizes and SHA-256 hashes in [`artifacts/manifest.json`](artifacts/manifest.json). They are excluded from Git history.

```bash
python3 tools/download_artifacts.py --list
python3 tools/download_artifacts.py --group data
python3 tools/download_artifacts.py --model compact_ia_joint --seed 42

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r environment/requirements-training.txt
python tools/evaluate.py --model compact_ia_joint --seed 42 --scope full
python tools/evaluate.py --model compact_ia_joint --seed 42 --scope filtered
```

The portable evaluator loads the fixed archived Ridge coefficients, uses float32 source embeddings and a float64 readout, and reports its replay difference on the original 2,000 functions. `--device cpu` is supported; CUDA is recommended for full evaluation. See [artifact availability](docs/ARTIFACTS.md) for which checkpoint downloads are currently present.

## Retrain and extend the experiments

```bash
python tools/build_caches.py
python tools/run_experiment.py joint --model IA --seed 42
python tools/run_experiment.py source_only --seed 42

# Prepare native token caches and the pinned public initializations:
python tools/prepare_pretrained.py
python tools/run_experiment.py pretrained --index 0 --dry-run
python tools/run_experiment.py pretrained --index 0
```

All final encoder-training comparisons use **9,762 updates**, effective batch size **256**, and seeds **42/43/44**. Pretrained indices 0–2 are CodeBERT IPC-only; 3–5 CodeBERT joint; 6–8 UniXcoder IPC-only; 9–11 UniXcoder joint. Validation-selected settings are frozen before test evaluation. The natural-language course extension is not part of the reported method.

Detailed commands, validation-search stages, compute requirements, and historical environment distinctions are in [REPRODUCING.md](docs/REPRODUCING.md).

## Package contents

| Path | Contents |
|---|---|
| `paper/` | Current manuscript, PDF, bibliography, figure sources |
| `configs/` | Frozen paper protocol and expanded-evaluation definition |
| `data/` | Tokenizers, split identities, probe indices, pretrained metadata |
| `results/` | Predictions, per-seed metrics, overlap witnesses, inference measurements |
| `experiments/` | Versioned training implementations, including the final four-condition pretrained study |
| `tools/` | Offline result reconstruction, downloads, evaluation, training dispatch, overlap audit, timing |
| `artifacts/manifest.json` | Downloadable files, hashes, and explicitly pending checkpoints |
| `environment/` | Dependency pins and observed GPU environment records |
| `provenance/` | Source snapshots, corpus hashes, package integrity manifest |
| `archive/` | Historical 30-minute results/documentation; not the current comparison |

The target is **LLVM-MCA 18.1.8 static IPC for Skylake, 100 iterations**, not measured runtime. Source data derive from AnghaBench; upstream datasets and pretrained models retain their respective terms. See [DATA_AND_LICENSING.md](docs/DATA_AND_LICENSING.md).

The [result map](docs/RESULT_MAP.md) links manuscript tables to files and commands. [VALIDATION.md](docs/VALIDATION.md) records what has actually been checked. Exploratory hardware measurements from earlier work are retained as supplementary provenance and are not substituted for the paper's static IPC labels.
