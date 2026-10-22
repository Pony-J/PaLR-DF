# Reproduction guide

Run commands from the repository root after installation. Outputs always go to explicitly named directories. Table and plot generation use only source-controlled records; checkpoint replay additionally requires the separate asset archive.

## 1. Verify the numerical records

```bash
palr-df verify-results --output runs/tables
palr-df figures --output runs/figures
```

Expected counts are 2,520 comparison rows (9 models × 2 horizons × 28 satellites × 5 seeds), 1,400 rank-intervention rows, and 560 spectral rows. The command checks hashes, duplicates, satellite/seed completeness, window counts, paper aggregate values and spectral medians. `runs/tables/verification.json` should report `passed: true` and reference differences at most 1e-12 ns. Figure generation preserves numerical data but uses portable fonts and can differ from manuscript typesetting.

## 2. Verify and replay weights

Extract `PaLR-DF-v0.1.0-assets.zip` into the repository root and check:

```bash
palr-df verify-assets --assets artifacts
palr-df evaluate --assets artifacts --family palr --horizon 12h --seed 2024 --satellites C06 C25 C29 --output runs/replay_example
```

The evaluator uses full-window batches, float32 forward computation and float64 clock reconstruction. The returned report compares each replay RMSE with the original archived metrics. Default absolute tolerances are 0.0002 ns on CPU and 0.000001 ns on CUDA. The full 560-state CPU replay measured a maximum difference of 0.0001296 ns from the original CUDA archive; the legacy and released code produce identical outputs on the same device in the checked cases. The per-window arrays allow direct inspection. Neither reselection of checkpoints nor retraining occurs during replay.

The included portable script runs the complete pretrained matrix:

```bash
python scripts/replay_paper.py --assets artifacts --output runs/full_replay --device cpu
```

Its equivalent loop is:

```python
import subprocess
import sys

for family in ("palr", "v2"):
    for horizon in ("12h", "24h"):
        for seed in range(2024, 2029):
            subprocess.run([
                sys.executable, "-m", "palr_df", "evaluate",
                "--assets", "artifacts", "--family", family,
                "--horizon", horizon, "--seed", str(seed),
                "--satellites", "all", "--device", "cpu",
                "--output", f"runs/replay/{family}/{horizon}/seed_{seed}",
            ], check=True)
```

This is a replay of the original historical 28-satellite result set. Keep any new experiment population and split separate from these archived paper results.

## 3. Recompute the weighted spectra

```bash
palr-df spectrum --assets artifacts --family v2 --horizon 12h --seed 2024 --satellites C29 --output runs/spectrum_v2_C29
palr-df spectrum --assets artifacts --family palr --horizon 24h --seed 2024 --satellites C29 --output runs/spectrum_palr_C29
```

Pass `--satellites all` for an entire seed/horizon. The command captures contexts on **training windows**, forms centered covariance and recomputes raw/context/task `r95` using the released loss weights. It compares ranks against the frozen records and saves singular values. Context collection uses batches of 64, matching the archived spectral analysis. Floating-point perturbations near an energy threshold can affect a reported rank; inspect singular values and runtime metadata in that case.

## 4. Train the compact model or adapters

```bash
palr-df train --config configs/palr_12h.json --data artifacts/data/code_clock.npz --satellite C29 --seed 2024 --device cpu --output runs/train_C29_12h
palr-df train --config configs/palr_24h.json --data artifacts/data/code_clock.npz --satellite C29 --seed 2024 --rank 12 --output runs/train_C29_24h_r12
palr-df train --config configs/baselines/cnn_12h.json --data artifacts/data/code_clock.npz --satellite C29 --seed 2024 --output runs/train_cnn_C29
```

Use the 12 baseline configs under `configs/baselines/` to select an adapter and horizon. To reproduce a training matrix, iterate over the required satellites, seeds and horizons, keeping separate output directories. The main manuscript recipe uses rank 3; the intervention ranks and seed allocations are specified in [protocol.md](protocol.md).

The fresh-training output includes `model.pt`, `config.json`, `history.json` and `report.json`. Add `--epochs 1` for a bounded implementation check; it is explicitly marked `implementation_smoke` and is not substituted for a paper result. The release was validated with PyTorch 2.5.1 and CUDA 11.8. Fresh training on another runtime is a reproduction attempt, while the model zoo provides the exact states used in the reported spectra and PaLR-DF/V2 comparisons.

## Runtime and expected effort

The tested scientific dependencies are pinned in `requirements.txt`. Python 3.11 is recommended; the source package declares Python 3.10–3.12. Source-only checks take seconds, and complete checkpoint replay is practical on CPU. GPU training is optional; use a PyTorch 2.5.1 CUDA build compatible with your system and pass `--device cuda`. Full training costs depend on the model, satellite and early-stopping epoch.

The release validation records measured wall time and numerical errors in [validation.json](../results/validation.json). CI runs the CPU mathematical/data checks, verifies frozen aggregates and regenerates the seven result plots without downloading large assets.
