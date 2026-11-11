# Validation record

Completed checks for this snapshot:

- Recomputed53 prediction files using a dependency-free metric implementation: original2,000-function controlled comparisons,21 full-test model/seed evaluations, feature references and the assembly-only reference.
- Verified identical function IDs and IPC labels across the model comparisons.
- Recomputed full-test and training-overlap-filtered R²/MAE/Spearman at all thresholds.8/.9/.95, matching recorded aggregates to1e-9.
- Original2,000 predictions exactly replayed during all21 expanded GPU evaluations (maximum absolute transformed-prediction difference0).
- All21 GPU evaluations completed successfully: compact models on A100/PyTorch2.5.1; final pretrained models on GH200/PyTorch2.9.1.
- Same-A100 float32 inference timing completed for three source architectures, with synchronization, explicit warmups/repetitions and allocated-memory records.
- Tokenizer normalization and complete prefix-retrieval logic checked, including exhaustive small-universe retrieval cases.
- Current manuscript compiled successfully:10pages including references; body ends within9pages.

Local CPU smoke checks also passed: 14 tokenizer regression cases, the compact encoder forward pass/parameter count, and frozen vocabulary/probe identities. Seven unit tests cover metric conventions and archive-installation guards.

The repository CI verifies package integrity, standard-library metric behavior, archive installation guards, syntax, and the full offline result reconstruction. It does not claim new full GPU retraining or universal hardware portability.

Fixed-model GPU replay and fresh retraining are different checks. Public final pretrained weight upload remains explicitly pending until the Arrhenius export can be retrieved. Historical mixed-view results are metric-level aggregates, not reconstructed per-function predictions.
