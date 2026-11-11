# Data lineage and attribution

The raw corpus contains AnghaBench-derived C functions, compiled assembly and recorded static IPC labels. The original partitions contain309,913 training,17,150 validation, and17,301 test functions. The paper's numerical target is the LLVM-MCA18.1.8 Skylake analysis result with100iterations. Raw-file hashes and the original code/cache provenance are retained.

AnghaBench is described by Da Silva et al., *AnghaBench: A Suite with One Million Compilable C Benchmarks for Code-Size Reduction*, CGO2021. The bibliography entry and the other upstream method citations are in `paper/references.bib`. Source functions originate in multiple upstream projects. This package does not replace their authorship or licensing terms, and no new license is assigned to third-party code or data.

CodeBERT and UniXcoder are upstream Microsoft pretrained models. Their original initialization commits, tokenizer metadata and configs are retained under `provenance/pretrained_initialization/` and `data/pretrained_metadata/`. Upstream model terms continue to apply. The downloader obtains initializations only through the explicit preparation command.

The current repository is an identified author replication snapshot. An author-owned-code redistribution license has not been selected on the author's behalf. Historical provenance paths may identify the original experimental machines; no anonymous-artifact claim is made.
