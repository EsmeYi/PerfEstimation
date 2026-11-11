# Artifact validation record

Author-side checks performed on 2 November 2026. No new training was launched during packaging.

- Recomputed 26 archived prediction files and verified identical 2,000 test IDs, order and labels. Recovered IA R² 0.7523 ± 0.0085 and BPE 0.7342 ± 0.0020; metrics match stored summaries.
- Recomputed hardware rank correlations from the archived measurement records, including the stable, length-filtered and branch-free subsets.
- Regenerated both prediction figures, retaining all 18,000 prediction pairs per figure and verifying common identities and bin inclusion.
- Passed 14 corrected-IA tokenizer regression checks, source encoder parameter count (7,517,696), finite forward pass, vocabulary size and frozen split identity checks.
- Re-encoded every evaluation input and matched the original 2,000 × 512 source, IA assembly and BPE assembly arrays exactly. Repeat with `python tools/validate_tokenization.py` after extracting data.
- Inventoried and hashed all three raw splits: 309,913 training, 17,150 original validation and 17,301 original test records. The evaluation protocol takes only the first 2,000 of each evaluation split. Repeat whole-file integrity checks with `python tools/verify_package.py --raw-data`.
- Loaded final seed-42 weights and ran finite CPU forward passes for IA, BPE, source-only, alignment-only, CodeBERT and UniXcoder. Pretrained native tokenizers and architecture configs load offline from the included metadata.
- All 18 recovered final checkpoint files are checked against SHA-256 values calculated during remote export. See `checkpoints/manifest.json`.
- Parsed all Python source files for syntax errors. Compiled the bundled manuscript with its three independent PDF figures.
- The release archive is extracted to a separate temporary directory and its integrity checker, result scripts, smoke check and both plotting scripts are run there before delivery.

The lightweight reconstruction route is validated end to end. Complete retraining and full 5,000+2,000-function inference from every checkpoint were not rerun during packaging. Cache reconstruction was verified for all test sequences; generating the complete training cache remains an explicit reproduction step. GPU training entry points preserve the historical algorithms and frozen settings, with documented path and output changes.

The archive has two historical coverage gaps: assembly-only final weights, and mixed-view final weights/per-function predictions were not originally saved. Their recorded results and training implementations are included. Evaluation-only exports omit optimizer and RNG state; use fresh run directories for training.
