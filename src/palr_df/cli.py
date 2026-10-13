"""Command line interfaces for the first-paper release."""
from __future__ import annotations
import argparse
import json
import platform
import sys
from pathlib import Path
from .io import read_json, safe_path, sha256, verify_manifest, write_json


def parser():
    root = argparse.ArgumentParser(prog="palr-df", description="PaLR-DF first-paper reproducibility tools")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("verify-results", "figures"):
        p = commands.add_parser(name)
        p.add_argument("--repo", type=Path, default=Path.cwd())
        p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("verify-assets")
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--repo", type=Path, default=Path.cwd())
    for name in ("evaluate", "spectrum"):
        p = commands.add_parser(name)
        p.add_argument("--assets", type=Path, required=True)
        p.add_argument("--repo", type=Path, default=Path.cwd())
        p.add_argument("--family", choices=["palr", "v2"], default="palr")
        p.add_argument("--horizon", choices=["12h", "24h"], default="12h")
        p.add_argument("--seed", type=int, default=2024)
        p.add_argument("--satellites", nargs="+", default=["C29"], help="Satellite identifiers, or all for the original 28-satellite archival replay")
        p.add_argument("--device", default="cpu")
        p.add_argument("--threads", type=int, default=4)
        p.add_argument("--output", type=Path, required=True)
        if name == "evaluate":
            p.add_argument("--atol-ns", type=float, help="Absolute replay tolerance; default CPU 0.0002 ns, CUDA 0.000001 ns")
    p = commands.add_parser("train")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--satellite", default="C29")
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--rank", type=int)
    p.add_argument("--epochs", type=int, help="Optional short implementation smoke run; recorded in config")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("prepare-data")
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--repo", type=Path, default=Path.cwd())
    p.add_argument("--output", type=Path, required=True)
    return root


def fresh_output(path):
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def runtime(torch, device):
    return {"python": platform.python_version(), "platform": platform.system(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "device": str(device), "threads": torch.get_num_threads()}


def replay(args):
    import numpy as np
    import pandas as pd
    import torch
    from .data import ClockData
    from .model import from_config
    from .train import seed_everything
    if args.command == "evaluate" and args.atol_ns is None:
        args.atol_ns = 2e-4 if args.device == "cpu" else 1e-6
    if args.threads < 1 or (args.command == "evaluate" and args.atol_ns < 0):
        raise ValueError("Threads must be positive and tolerance nonnegative")
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    fresh_output(args.output)
    # The source-controlled manifest is the trust anchor for the release assets.
    verify_manifest(args.assets, args.repo / "results/frozen/assets_manifest.json")
    index = read_json(args.repo / "results/frozen/checkpoint_index.json")
    selected = [r for r in index if r["family"] == args.family and r["horizon"] == args.horizon and r["seed"] == args.seed]
    requested = set(args.satellites)
    if requested == {"all"}:
        requested = {r["satellite"] for r in selected}
    selected = [r for r in selected if r["satellite"] in requested]
    if len(selected) != len(requested):
        raise ValueError("Requested satellite/seed checkpoints are unavailable")
    data = ClockData(args.assets / "data/code_clock.npz")
    frozen = pd.read_csv(args.repo / "results/frozen/comparison.csv")
    spectral = pd.read_csv(args.repo / "results/frozen/spectra.csv")
    model_name = "PaLR-DF" if args.family == "palr" else "V2"
    reports = []
    for entry in selected:
        config = read_json(safe_path(args.assets, entry["config"]))
        model = from_config(config)
        model.load_state_dict(torch.load(safe_path(args.assets, entry["checkpoint"]), map_location="cpu", weights_only=True), strict=True)
        split = "validation" if args.command == "evaluate" else "train"
        samples = data.samples(config, entry["satellite"], split)
        if args.command == "evaluate":
            from .evaluate import evaluate
            summary, windows, differences, predicted = evaluate(model, samples, args.device)
            row = frozen[(frozen.horizon == args.horizon) & (frozen.model == model_name) & (frozen.seed == args.seed) & (frozen.satellite == entry["satellite"])].iloc[0]
            error = abs(summary["rmse_ns"] - float(row.rmse_ns))
            report = {"satellite": entry["satellite"], "n_windows": len(samples), **summary, "reference_rmse_ns": float(row.rmse_ns), "absolute_error_ns": error, "atol_ns": args.atol_ns, "passed": error <= args.atol_ns}
            pd.DataFrame({"target_start_gpst": samples.target_start.astype(str), **windows}).to_csv(args.output / f"{entry['satellite']}_windows.csv", index=False)
            np.savez_compressed(args.output / f"{entry['satellite']}_predictions.npz", target_start_gpst=samples.target_start, predicted_difference_ns=differences, predicted_bias_ns=predicted, target_bias_ns=samples.target_bias_ns)
        else:
            from .spectra import diagnose
            report = {"satellite": entry["satellite"], **diagnose(model, samples, config["training"], args.device)}
            row = spectral[(spectral.horizon == args.horizon) & (spectral.model == model_name) & (spectral.seed == args.seed) & (spectral.satellite == entry["satellite"])].iloc[0]
            report["reference_r95"] = {k: int(row[k]) for k in ("raw_r95", "context_r95", "task_r95")}
            report["passed"] = all(report[k] == report["reference_r95"][k] for k in report["reference_r95"])
        report["checkpoint_sha256"] = entry["sha256"]
        reports.append(report)
        print(f"{entry['satellite']}: passed={report['passed']}", flush=True)
    result = {"passed": all(r["passed"] for r in reports), "mode": "archival_checkpoint_replay", "family": args.family, "horizon": args.horizon, "seed": args.seed, "runtime": runtime(torch, args.device), "records": reports}
    write_json(args.output / "report.json", result)
    if not result["passed"]:
        raise ValueError(f"Replay differs from reference; inspect {args.output / 'report.json'}")
    return {"passed": True, "checkpoint_count": len(reports)}


def training(args):
    import torch
    from .data import ClockData
    from .model import from_config
    from .train import seed_everything, initialize_pca, fit
    from .evaluate import evaluate
    if args.threads < 1:
        raise ValueError("Threads must be positive")
    torch.set_num_threads(args.threads)
    config = read_json(args.config)
    if args.satellite not in config["satellites"]:
        raise ValueError("Satellite is absent from the selected recipe")
    config["satellites"] = [args.satellite]
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("epochs must be positive")
        config["training"]["max_epochs"] = args.epochs
    if args.rank is not None:
        if args.rank < 1:
            raise ValueError("rank must be positive")
        config["model"]["basis_count"] = args.rank
    config["training"]["seed"] = args.seed
    config["run_satellite"] = args.satellite
    config["run_kind"] = "implementation_smoke" if args.epochs is not None else "retraining"
    fresh_output(args.output)
    seed_everything(args.seed)
    data = ClockData(args.data)
    train = data.samples(config, args.satellite, "train")
    validation = data.samples(config, args.satellite, "validation")
    model = from_config(config)
    if config["training"].get("shape_basis_initialization", "random") == "train_pca":
        initialize_pca(model, train)
    best_epoch, history = fit(model, train, validation, config, args.device)
    checkpoint = args.output / "model.pt"
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, checkpoint)
    write_json(args.output / "config.json", config)
    write_json(args.output / "history.json", history)
    metrics = evaluate(model, validation, args.device)[0]
    result = {"satellite": args.satellite, "seed": args.seed, "train_windows": len(train), "validation_windows": len(validation), "best_epoch": best_epoch, "metrics": metrics, "checkpoint_sha256": sha256(checkpoint), "data_sha256": sha256(args.data), "source_config_sha256": sha256(args.config), "runtime": runtime(torch, args.device), "run_kind": config["run_kind"]}
    write_json(args.output / "report.json", result)
    return result


def main():
    args = parser().parse_args()
    try:
        if args.command == "verify-assets":
            result = verify_manifest(args.assets, args.repo / "results/frozen/assets_manifest.json")
        elif args.command in {"verify-results", "figures"}:
            from .results import verify_results, figures
            result = (verify_results if args.command == "verify-results" else figures)(args.repo, args.output)
        elif args.command in {"evaluate", "spectrum"}:
            result = replay(args)
        elif args.command == "train":
            result = training(args)
        else:
            from .prepare import prepare_data
            frozen = args.repo / "results/frozen"
            result = prepare_data(args.raw_dir, frozen / "raw_products.json", read_json(frozen / "provenance.json")["satellites"], args.output)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as error:
        print(f"palr-df: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
