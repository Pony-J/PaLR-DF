# Model zoo

The v0.1.0 asset archive contains all checkpoints for **28 satellites × 5 seeds × 2 horizons × 2 families = 560**. Each is a plain tensor state dict, loaded with `weights_only=True` and strict key matching. There is no pickled training object or embedded executable model class.

| Family | Horizon | Residual rank | Context width | Parameters |
|---|---:|---:|---:|---:|
| PaLR-DF | 12 h | 3 | 64 | 33,019 |
| PaLR-DF | 24 h | 3 | 64 | 37,339 |
| V2 | 12 h | 48 | 128 | 149,044 |
| V2 | 24 h | 48 | 128 | 218,164 |

Paths follow `checkpoints/{palr|v2}/{12h|24h}/seed_{2024..2028}/{satellite}.pt`. Matching portable configs are under `configs/{family}/{horizon}/seed_{seed}.json`. [checkpoint_index.json](../results/frozen/checkpoint_index.json) maps every checkpoint to its hash and config, retaining the original config's hash for provenance. Config cleanup removes machine-specific data roots and unused experiment fields; model and training settings are preserved.

Pretrained rank-intervention and comparator weights are outside this focused model zoo. Complete frozen result grids remain in the source repository, and PaLR-DF rank variants and local comparison adapters can be trained using the supplied code and recipes. See [reproduction.md](reproduction.md) for what each command verifies.
