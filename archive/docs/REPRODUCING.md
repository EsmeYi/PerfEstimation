# Reproduction guide

All commands below run at the package root. Use a fresh package copy for retraining; archived `results/` are reference outputs and new runs write beneath `experiments/`. No script in the quick-start launches training.

## 1. Prepare the environment and caches

The observed GPU environment used Python 3.11 and CUDA PyTorch 2.5.1. Install a CUDA build suitable for your machine, then the versions recorded in `environment/requirements-training.txt`. The observations were collected at export time rather than a package freeze for every historical run. Scientific kernels can differ slightly across GPU/software versions.

```sh
python3.11 -m venv .venv-training
. .venv-training/bin/activate
python -m pip install -r environment/requirements-training.txt
python tools/smoke_check.py
tar -xzf /path/to/ipc-data.tar.gz
python tools/build_caches.py
```

`build_caches.py` regenerates source BPE, assembly BPE and corrected IA arrays. It preserves original order, checks training/test raw-byte fingerprints, and verifies the frozen probe and evaluation IDs. It uses all 309,913 training records and the first 2,000 records of each evaluation partition. Raw JSONL already contains source, assembly and labels; this step does not compile functions or regenerate labels. Derived caches are overwritten if explicitly rerun.

To use an existing raw dataset instead, pass `--raw-dir /path/to/jsonl-directory` (also accepts `.jsonl.gz`).

## 2. Evaluate recovered compact model weights

```sh
tar -xf /path/to/ipc-checkpoints.tar
python tools/verify_package.py --checkpoints
python tools/evaluate_compact.py --checkpoint checkpoints/IA_seed42.pt --device cuda
```

Replace the checkpoint with BPE, source-only or alignment-only exports as desired. The evaluator rebuilds the readout from the same 5,000 training identities and saves its coefficients, test predictions and metrics under `generated/checkpoint_evaluation/`. Use `--output` to retain multiple evaluations. CPU is supported via `--device cpu`, but full encoding will be much slower. Optional `--modality assembly` evaluates a joint model's assembly encoder; this is distinct from the separately trained assembly-only reference.

## 3. Evaluate recovered pretrained weights offline

After building common caches:

```sh
python tools/prepare_pretrained.py --tokenize-only
python tools/evaluate_pretrained.py --model codebert --seed 42
python tools/evaluate_pretrained.py --model unixcoder --seed 42
```

Native tokenization is reconstructed from archived metadata, and final networks are instantiated from archived configs and exported weights. No pretrained initialization download is needed for this route. Evaluation uses CUDA and the historical 768-dimensional masked-mean embedding followed by the same fixed Ridge protocol. UniXcoder uses its encoder-only sequence format.

## 4. Retrain the compact models and ablations

The dispatcher's `--dry-run` prints exact commands without running them:

```sh
python tools/run_experiment.py joint --model IA --seed 42 --dry-run
python tools/run_experiment.py joint --model IA --seed 42
python tools/run_experiment.py joint --model BPE --seed 42
python tools/run_experiment.py source_only --seed 42
python tools/run_experiment.py alignment_only --seed 42
python tools/run_experiment.py asm_only --seed 42
python tools/run_experiment.py mixed_view --seed 42
python tools/run_experiment.py features
```

Repeat seeded commands for 43 and 44. The joint and supervision dispatchers run the portable test evaluator after training and write `replicated_evaluation/` beside the new checkpoint. Frozen IA/BPE settings are taken from the recorded search/promotion and final configuration files. Seed 42 uses the selected full-stage implementation; in a fresh artifact it trains from initialization, whereas a historical screening checkpoint, when present, is resumed. No screening checkpoints are distributed. These are recipe replications rather than bit-for-bit reconstructions of that historical screen/resume boundary.

Mixed-view training uses a 0.5 probability of removing available declaration context while preserving the original label. Its supplementary prepare step is run automatically. Assembly-only trains an IA assembly encoder directly against IPC. These scripts now save final weights, and mixed-view also saves per-function predictions. Archived metrics are never replaced.

`tools/install_checkpoints.py` is optional and only needed by original supplementary scripts that expect historical checkpoint paths. It creates symlinks to evaluation-only exports. Do not use these as training resume states. The dispatcher rejects retraining while these links are installed; use a fresh package copy for training.

## 5. Retrain CodeBERT and UniXcoder

Initializations are separate from the exported final weights. To download them and convert to safetensors, use a separate preparation environment with PyTorch >=2.6 (required by Transformers when reading upstream PyTorch `.bin` files), plus the recorded Transformers, NumPy, tokenizers and safetensors versions. Public network access is required. The recorded original initialization commit is used when available.

```sh
python tools/prepare_pretrained.py
```

Return to the training environment, then:

```sh
python tools/run_experiment.py pretrained --index 0
```

Indices 0–2 are CodeBERT seeds 42–44; 3–5 are UniXcoder seeds 42–44. Each final run uses 1,800 seconds of training wall time, excluding model loading, tokenization and evaluation; training periodic checkpoint time is included. Frozen selected LR is 5e-5. The original entry point also supports the validation-only LR search: `python experiments/pretrained_ipc_30min_20261028/experiment.py train --index N`, where N=0–5 covers the two models and three learning rates; `select` writes the selected configuration. Keep reference `selected.json` unchanged when reproducing final runs.

## 6. Supplementary cost, robustness and hardware experiments

Archived results can be inspected and reaggregated without hardware counters. To run the original supplementary implementations:

```sh
python tools/install_checkpoints.py
python tools/materialize_predictions.py
export LLVM_BIN=/absolute/path/to/llvm-18.1.8/bin
python tools/run_experiment.py speed
```

Cost measurements depend on CPU/GPU, batch size, synchronization and timing boundaries. Compare the stored `speed_robustness.json` for the historical setup. Retain new outputs separately when comparing runs.

Hardware measurement requires Linux, LLVM 18.1.8, GCC, taskset, accessible user-mode hardware counters, and an allocated CPU core on which to run synthetic function harnesses. It runs generated native C code and can take appreciable time; use an appropriate isolated compute allocation rather than a shared login node. The source selection and signature filters are in `hw_prepare.py`, with per-function timeouts and fallback handling in `run_hw.py`/`harness.c.in`.

```sh
python tools/run_experiment.py hardware_prepare
python tools/run_experiment.py hardware_measure
python tools/run_experiment.py hardware_predict
python tools/run_experiment.py hardware_analyze
```

For archived measurements, prefer `python tools/reproduce_hardware_summary.py`; no perf permissions or native execution are needed. Measurements on another microarchitecture produce a new hardware dataset, not an exact repeat of the EPYC 9534 measurements. See `results/supplement/hardware_environment.txt` and `hw_environment.txt`.
