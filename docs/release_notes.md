# v0.1.0 release notes

First-paper release of **PaLR-DF**, supporting the manuscript *Long-Horizon Satellite Clock Bias Prediction: From Mapping Redundancy to Low-Rank Decoding*.

## Assets

| File | Contents |
|---|---|
| `PaLR-DF-v0.1.0-source.zip` | Standalone repository, frozen numerical records, guides and tests |
| `PaLR-DF-v0.1.0-assets.zip` | Portable CODE clock-data extract, matching configs, 560 PaLR-DF/V2 states and hashes |
| `palr_df-0.1.0-py3-none-any.whl` | Optional offline installation of the Python package; use with the source repository's configs and records |
| `RELEASE_SHA256SUMS.txt` | Checksums of the three release files |

Extract the asset archive into the source repository to obtain `artifacts/`. Follow the README for the pinned runtime, table verification, checkpoint replay and spectral recomputation. Code uses MIT; CODE/IGS data retains provider terms.

## Verified behavior

The release rebuilds the complete comparison and rank aggregates, replays all 560 pretrained states on CPU and CUDA, reconstructs the processed data from the manifest-listed raw products, and checks forward/window agreement with the archived implementation. The 12 CPU tests cover alignment, integrity, effective-rank mathematics, metric aggregation, model geometry and comparison adapters. Machine-readable measurements are recorded in `results/validation.json`.

The model zoo covers the core PaLR-DF and V2 families. Selected comparison-adapter recipes and complete frozen comparison records are supplied separately from pretrained weights. The historical evaluation protocol is documented in `docs/protocol.md`.
