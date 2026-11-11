# Data and model availability

Downloadable assets, their compressed sizes, and their compressed/uncompressed SHA-256 hashes are in [`artifacts/manifest.json`](../artifacts/manifest.json). The downloader verifies bytes before installing them. Large files are Release assets, not Git objects.

| Artifact | Available evidence | Weight availability |
|---|---|---|
| Raw train/validation/test corpus | Complete JSONL split archive | `ipc-data.tar.gz` |
| Compact IA joint, BPE joint, IPC-only, alignment-only | Three-seed reference metrics and predictions | 12 evaluation-only checkpoints |
| Final CodeBERT IPC-only/joint, UniXcoder IPC-only/joint | All12 final original-sample runs and full-test predictions; coefficients and selected configs | Explicitly listed under `pending_assets`: Arrhenius export requested; login unavailable before retrieval |
| Assembly-only reference | Three-seed predictions and metrics | Historical checkpoints not retained; training recipe supplied |
| Mixed-view declaration adaptation | Historical three-seed metrics | Historical checkpoints and per-function outputs not retained; retraining recipe supplied |

The public pretrained initializations remain available from their original providers and are pinned by commit in this package. They are separate from task-adapted final weights. The complete final training code and data allow retraining when a final checkpoint is not yet downloadable.

Evaluation exports omit optimizer state, sampler position, and RNG state. Do not resume training from them. Run fresh training in a separate package copy. No historical30-minute pretrained export is advertised as a final9,762-update checkpoint.

All reported full-test predictions and fixed Ridge readouts are tracked in Git, so verifying the results does not depend on any pending model upload.
