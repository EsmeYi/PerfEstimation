# Reproduction guide

Run commands at the repository root. New outputs go under ignored `generated/` or experiment run directories; reference `results/` remain unchanged.

## 1. Offline result reconstruction

```bash
python3 tools/verify_package.py
python3 tools/reproduce_results.py
```

This is the fastest replication route: only Python's standard library is used. The script recomputes 53 prediction files, checks shared original/full-test IDs and labels, recomputes all three overlap-filter thresholds, and verifies the archived expanded aggregates to 1e-9. The mixed-view table is reaggregated from historical per-seed metrics because those runs did not save per-example predictions.

To regenerate the supervision-ablation prediction figure:

```bash
python -m pip install -r environment/requirements-figures.txt
python tools/plot_ablation_predictions.py
```

To compile the paper:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## 2. Data, checkpoints, and dependencies

```bash
python3 tools/download_artifacts.py --list
python3 tools/download_artifacts.py --group data
python3 tools/download_artifacts.py --group compact
```

Downloads and decompressed checkpoint files are checked against SHA-256. The corpus creates `data/raw/{train,val,test}.jsonl`, approximately 3 GB in total. Model archives preserve the saved tensor precision and omit optimizer/RNG state; they are for evaluation, not training resume. [ARTIFACTS.md](ARTIFACTS.md) records checkpoint coverage.

For CUDA training/evaluation, install a matching PyTorch build and the versions in `environment/requirements-training.txt`. The final pretrained study used Python 3.13.5, PyTorch 2.9.1/CUDA 12.9 on GH200. Historical compact training and the reference timing run used PyTorch 2.5.1 on A100. Exact observed records are in `environment/`; the current dependency file targets the final trainer. GPU/software versions can change floating-point results and timing.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r environment/requirements-training.txt
python tools/smoke_check.py
```

Raw JSONL already contains source, assembly, and IPC. Cache preparation does not compile functions or regenerate analyzer labels.

## 3. Frozen-model evaluation

```bash
python tools/evaluate.py --model compact_ia_joint --seed 42 --scope original
python tools/evaluate.py --model compact_ia_joint --seed 42 --scope full
python tools/evaluate.py --model compact_ia_joint --seed 42 --scope filtered
```

`--scope original` evaluates the original first2,000 test functions, `full` all17,301, and `filtered` the fixed12,600-function overlap-filtered subset. All use the same archived Ridge coefficients. Source tokenization is cached under `generated/full_cache/` with raw-file and tokenizer fingerprints. Use `--output` to retain multiple evaluations, `--device cpu` for CPU inference, or `--limit 2` for a short forward check.

Models accepted by this evaluator are `compact_ipc`, `compact_bpe_joint`, `compact_ia_joint`, `codebert_ipc`, `codebert_joint`, `unixcoder_ipc`, and `unixcoder_joint`. For the final pretrained checkpoints, download the matching model/seed only when listed as available by the manifest; otherwise retrain using section5. No historical30-minute checkpoint is used as a replacement.

The older `tools/evaluate_compact.py` additionally supports the archived alignment-only encoder and assembly branch. It requires rebuilt training caches and refits the original5,000-function Ridge:

```bash
python tools/build_caches.py
python tools/evaluate_compact.py --checkpoint checkpoints/alignment_only_seed42.pt --device cuda
```

## 4. Retrain compact models and reference models

Use a fresh working copy without installed evaluation-only checkpoint links. First rebuild the original runtime arrays:

```bash
python tools/build_caches.py
python tools/run_experiment.py joint --model IA --seed 42 --dry-run
python tools/run_experiment.py joint --model IA --seed 42
python tools/run_experiment.py joint --model BPE --seed 42
python tools/run_experiment.py source_only --seed 42
python tools/run_experiment.py alignment_only --seed 42
python tools/run_experiment.py asm_only --seed 42
python tools/run_experiment.py mixed_view --seed 42
python tools/run_experiment.py features
```

Repeat seeded commands for43/44. IA/BPE use their frozen selected target and weight. The seed42 joint recipe uses the selected full-stage implementation; a fresh package trains from initialization, whereas the original screening run continued from its saved2,000-step checkpoint. The shared step-based schedule and sampled data order are specified; no historical screening resume state is distributed.

Mixed-view training uses replacement probability0.5 and preserves the original implementation labels. Historical mixed-view and assembly-only evaluation weights were not retained. Their archived metrics/predictions are available as described in ARTIFACTS.md; the supplied training implementations now save final weights, and mixed-view also exports predictions for new runs.

## 5. Final fixed-update pretrained study

Prepare native token caches and download the pinned original public initializations:

```bash
python tools/prepare_pretrained.py
```

Original public commits are recorded under `provenance/pretrained_initialization/`; model configs and native tokenizers are archived. Legacy `vocab.json`/`merges.txt` are losslessly derived from each archived `tokenizer.json` for compatibility with older Transformers versions; offline smoke checks compare their token IDs. Initialization acquisition requires network access and PyTorch>=2.6 when loading upstream PyTorch binary weights. The supplied2.9.1 environment supports this route. `--tokenize-only` skips initialization downloads for offline evaluation/cache preparation.

```bash
python tools/run_experiment.py pretrained --index 0 --dry-run
python tools/run_experiment.py pretrained --index 0
```

| Index | Encoder | Condition | Seeds |
|---|---|---|---|
| 0–2 | CodeBERT | IPC-only | 42,43,44 |
| 3–5 | CodeBERT | Joint | 42,43,44 |
| 6–8 | UniXcoder | IPC-only | 42,43,44 |
| 9–11 | UniXcoder | Joint | 42,43,44 |

Each final run uses9,762 optimizer updates and effective batch256 through microbatches16. Both conditions use full-batch gradient caching with dropout replay, the same source architecture, full768-dimensional mean-pooled representation, and source learning rate5e-5. Joint training adds the corrected IA assembly encoder and a training-only768→128 projection. Its loss is MSE+γInfoNCE, with γ=.03 for CodeBERT and γ=1/5.990544826588272 for UniXcoder. The projection and assembly branch are omitted from the final source predictor.

The full validation-search implementation is retained in `experiments/pretrained_equal_steps_20261108/experiment.py`. Stages are `ipc_screen` (indices0–5), `ipc_rank`, `ipc_promote` (0–3), `ipc_select`, `joint_screen` (0–5), `joint_rank`, `joint_promote` (0–3), `joint_select`, `final` (0–11), and `aggregate`. Run search stages in that order in a separate copy if repeating selection. Final evaluation reuses frozen selected settings; no test-based seed/configuration choice is made.

## 6. Training-overlap audit and inference measurement

```bash
python tools/deduplicate.py --check
python tools/deduplicate.py --raw-dir data/raw --output generated/dedup
python tools/benchmark_inference.py --models compact_ia_joint
# Once final pretrained checkpoints are downloaded or retrained:
python tools/benchmark_inference.py
```

The overlap audit streams all309,913 training functions against all17,301 test functions. It uses normalized lexical token5-gram set Jaccard, complete prefix-filter candidate retrieval at threshold.8, exact hashed-shingle intersections, and thresholds.8/.9/.95. It exports witnesses and masks without consulting model predictions. At.9,12,600 full-test and1,398 original-sample functions remain. This sensitivity analysis is not a project-disjoint train/test construction.

Timing uses one physical GPU for all requested architectures, float32, length512, batch sizes1/16,20warmups and100synchronized repetitions. It times source encoding plus a dimension-matched scalar linear readout, records PyTorch allocated/reserved peaks, and measures host tokenization separately. Reference timings used archived earlier pretrained weights with the same source architectures; the portable timing command uses the final weights when available. Timings will depend on hardware/runtime.

## 7. Cluster submission

The portable entry points contain no required account names or cluster paths. Place the commands above inside your cluster's batch script, requesting one CUDA GPU and suitable CPU/memory/time resources. The final training's logical batch256 need not fit simultaneously in memory because the trainer uses microbatches16. Cache generation and overlap auditing are CPU tasks; inference uses float32. Set `HF_HOME` and `MPLCONFIGDIR` to writable cache directories if needed.

Historical30-minute experiments and exploratory hardware measurements are retained for provenance. They are not the current fixed-update comparisons or the static target's validation set.
