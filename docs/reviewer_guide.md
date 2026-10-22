# Guide to the paper artifacts

This repository accompanies *Long-Horizon Satellite Clock Bias Prediction: From Mapping Redundancy to Low-Rank Decoding*, by Yuanjin Bao, Jifei Tang and Lanhua Xia. This guide uses the figure and table numbering of the October 22, 2026 submission.

## A short verification route

After the [README installation](../README.md#quick-start), run from the repository root:

```bash
python -m unittest discover -s tests -v
palr-df verify-results --output runs/tables
palr-df figures --output runs/figures
```

Expected verification: `passed: true`, 2,520 comparison rows, 1,400 rank-intervention rows, 560 spectral records and a maximum aggregate reference error below 1e-12 ns. Figure generation writes the seven result figures in PNG, PDF and SVG, plus `figure_manifest.json`, which records the paper figure number, plotted values, layout and output hashes. The original manuscript plot PNGs are in [docs/assets](assets/); regeneration uses Times New Roman when available and a portable serif fallback otherwise.

For a model-level check, download the [v0.1.0 assets](https://github.com/Pony-J/PaLR-DF/releases/download/v0.1.0/PaLR-DF-v0.1.0-assets.zip) (244,422,396 bytes), verify [the release checksums](https://github.com/Pony-J/PaLR-DF/releases/download/v0.1.0/RELEASE_SHA256SUMS.txt), and extract into the repository root to obtain `artifacts/`.

```bash
palr-df verify-assets --assets artifacts
palr-df evaluate --assets artifacts --family palr --horizon 12h --seed 2024 --satellites C29 --output runs/replay_C29
palr-df spectrum --assets artifacts --family v2 --horizon 12h --seed 2024 --satellites C29 --output runs/spectrum_C29
```

The asset verifier checks 582 files, including all 560 PaLR-DF/V2 checkpoints. CPU replay tolerance is 0.0002 ns; the original archived values were obtained on CUDA. Full matrix replay and fresh training instructions are in [reproduction.md](reproduction.md).

## Paper-to-file index

| Submitted paper | Released evidence | Verification |
|---|---|---|
| Fig. 1: research overview | `docs/assets/research_overview.jpeg` | Original submitted image extracted without redrawing |
| Table I; architecture and parameter formula | `src/palr_df/model.py`, `configs/palr_12h.json`, `configs/palr_24h.json` | CPU tests; `train`; strict checkpoint replay |
| Figs. 2 and 5: ordinary/context/task-weighted spectra | `results/frozen/spectra.csv`; V2 and PaLR-DF weights | `verify-results`, `spectrum`, `figures` |
| Fig. 6: decoder-rank intervention | `rank_intervention.csv`, `rank_reference_3seed.csv`, `rank_reference_5seed.csv` | `verify-results`, `figures`; `train --rank` for fresh runs |
| Tables III and VI: 12 h / 24 h RMSE comparisons | `comparison.csv`, `paper_reference.csv` | `verify-results`; PaLR-DF/V2 `evaluate` |
| Fig. 7: paired satellite comparisons | `paired_12h_with_MGAMNet.csv` and complete comparison grid | `figures`; archived confidence intervals and tests retained |
| Fig. 8; parameter/RMSE columns of Tables IV–V | `complexity_points.csv`, `comparison.csv`; model configs | `figures`, `verify-results`; CPU parameter-count tests |
| Fig. 9: PaLR-DF minus INT-Best, all 28 satellites | `satellite_differences.csv` | `figures`; current single-panel format |
| Fig. 10: historical C41 daily diagnostic | `c41_daily.csv` | `figures`; archived daily values, without new inference |
| Table A3: data and training settings | `configs/`, [data card](data.md), [protocol](protocol.md) | Data alignment tests; asset verification; training recipe |

CSV paths in this table are relative to `results/frozen/`. `verify-results` also writes `comparison_paper_names.csv` and `comparison_distribution.csv` for inspecting manuscript display names and satellite error quantiles. Frozen source identifiers and numerical files remain unchanged.

## Scope and interpretation

The released model zoo covers the core PaLR-DF and V2 experiments. The January comparison block is used for checkpoint selection and the reported development evaluation, as described in Section V-A and [the protocol](protocol.md). INT-Best is the satellite-specific internal selection reference. The original 28-satellite, five-seed records are preserved in full.

The local comparator code and configurations are supplied for inspection and retraining; comparator and rank-sweep pretrained weights are outside this model zoo. Informer is the manuscript display name for the frozen CSV identifier `Informer-D`. [Comparison implementation details](protocol.md#comparison-implementations) document the exact adapters and input crops.

The supplementary experiments in Tables II, VII–IX and A1–A2 have separate experimental scopes and are not reproduced by the commands above. In particular, the released core model requires at least 24 h of history; the short-history extension used in Table VIII is separate from this model zoo. This guide does not substitute main-experiment records for those supplementary results.

The current `main` branch and [v0.1.1 source release](https://github.com/Pony-J/PaLR-DF/releases/tag/v0.1.1) provide the updated figure and navigation code and remain compatible with the immutable v0.1.0 data/checkpoint archive. The [v0.1.0 release](https://github.com/Pony-J/PaLR-DF/releases/tag/v0.1.0) preserves the original software snapshot and measured release validation.
