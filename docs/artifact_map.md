# Paper-to-artifact map

For figure/table numbers in the October 22 submission and a short verification route, see the [review guide](reviewer_guide.md).

| Paper content | Included evidence | Verification |
|---|---|---|
| PaLR-DF architecture and rank-3 decoder | `src/palr_df/model.py`, configs, all 280 PaLR-DF weights | `evaluate`, `train`; strict state loading and archived-forward parity checked for release |
| Ordinary/context/task weighted spectra | `spectra.py`, 560 checkpoint-level records, all 280 V2 reference weights | `spectrum` from training contexts; `verify-results` for frozen median statistics |
| Rank intervention curves | 1,400 per-satellite/seed rows, separate 3-/5-seed references | `verify-results`, `figures`; `train --rank` for fresh runs |
| 12 h and 24 h comparison tables | 2,520 per-satellite/seed rows, source-metric hashes | `verify-results`; PaLR-DF/V2 checkpoint replay via `evaluate` |
| Accuracy–parameter tradeoff | `complexity_points.csv`, parameter counts and comparison records | `figures`; model parameter counts checked in CPU tests |
| Paired satellite comparisons | Frozen paired statistics and satellite differences, with all satellites retained | Source records for inspection; aggregates rebuilt from `comparison.csv` |
| C41 diagnostic | Frozen daily and satellite-level records | Included as historical first-paper data |
| Original research-logic overview | `docs/assets/research_overview.png` | Original manuscript image retained |

The release focuses on the core first-paper contribution. It includes selected comparison-adapter code and configs; pretrained comparator and rank-sweep weights, every reference-policy branch, and auxiliary appendix experiments are outside this model zoo. The frozen tables clearly retain all selected experiments' satellites and available seeds.

`results/frozen_manifest.json` records source-controlled frozen file hashes. `results/frozen/assets_manifest.json` is the source-controlled trust anchor for the external data/checkpoint package. The checkpoint index retains original checkpoint and config hashes. Local machine paths are removed from the public package; logical asset paths replace them without changing checkpoint bytes or numerical records.

Reference aggregation values were exported independently from the manuscript's archived statistics and completed 24 h comparison table. The release verifier rebuilds them from the underlying satellite/seed rows. It also checks rank-intervention summaries against the independent archived three- and five-seed tables.

`figures` regenerates Figs. 2, 5–10, including the current single-panel Fig. 9, from these frozen files. Manuscript display names are applied at presentation time; the source records, model identifiers and hashes retain their release values.
