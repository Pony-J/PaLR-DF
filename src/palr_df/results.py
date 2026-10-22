"""Rebuild paper aggregates from complete satellite/seed records."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .io import read_json, verify_manifest, write_json


PAPER_NAMES = {"Informer-D": "Informer"}


def checked_records(frame, group_columns, satellites, seeds):
    keys = group_columns + ["satellite", "seed"]
    if frame.duplicated(keys).any() or not np.isfinite(frame["rmse_ns"]).all():
        raise ValueError("Duplicate or non-finite result records")
    for _, block in frame.groupby(group_columns):
        if set(block["seed"]) != set(seeds):
            raise ValueError("Missing seed in comparison block")
        for _, local in block.groupby("seed"):
            if len(local) != len(satellites) or set(local["satellite"]) != set(satellites):
                raise ValueError("Incomplete satellite set")


def comparison_summary(frame):
    seed_macro = frame.groupby(["horizon", "model", "seed"], as_index=False)["rmse_ns"].mean()
    summary = seed_macro.groupby(["horizon", "model"])["rmse_ns"].agg(["mean", "std"]).reset_index()
    return seed_macro, summary.rename(columns={"mean": "macro_rmse_ns", "std": "sample_sd_ns"})


def rank_summary(frame, seeds):
    local = frame[frame.seed.isin(seeds)]
    groups = local.groupby(["horizon", "rank", "seed"], as_index=False)["rmse_ns"].mean()
    valid = groups.groupby(["horizon", "rank"])["seed"].nunique()
    valid = valid[valid == len(seeds)].index
    out = groups.groupby(["horizon", "rank"])["rmse_ns"].agg(["mean", "std"]).loc[valid].reset_index()
    return out.rename(columns={"mean": "macro_rmse_ns", "std": "sample_sd_ns"})


def verify_results(repo, output):
    repo, output = Path(repo), Path(output)
    integrity = verify_manifest(repo, repo / "results/frozen_manifest.json")
    frozen = repo / "results/frozen"
    provenance = read_json(frozen / "provenance.json")
    records = pd.read_csv(frozen / "comparison.csv")
    checked_records(records, ["horizon", "model"], provenance["satellites"], provenance["seeds"])
    if len(records) != 2520 or records.model.nunique() != 9 or set(records.horizon) != {"12h", "24h"}:
        raise ValueError("Unexpected comparison grid")
    if not (records.n_windows == records.horizon.map({"12h": 42, "24h": 36})).all():
        raise ValueError("Unexpected evaluation window count")
    seed_macro, table = comparison_summary(records)
    expected = pd.read_csv(frozen / "paper_reference.csv")
    join = table.merge(expected, on=["horizon", "model"], suffixes=("", "_reference"), validate="one_to_one")
    if len(join) != len(table):
        raise ValueError("Reference table is incomplete")
    max_error = max(float(np.abs(join[c] - join[c + "_reference"]).max()) for c in ("macro_rmse_ns", "sample_sd_ns"))
    if max_error > 1e-12:
        raise ValueError(f"Paper comparison mismatch: {max_error}")
    ranks = pd.read_csv(frozen / "rank_intervention.csv")
    if ranks.duplicated(["horizon", "rank", "satellite", "seed"]).any() or len(ranks) != 1400:
        raise ValueError("Unexpected rank grid")
    for (_, _, _), group in ranks.groupby(["horizon", "rank", "seed"]):
        if len(group) != 28 or set(group.satellite) != set(provenance["satellites"]):
            raise ValueError("Incomplete rank-intervention satellite set")
    for n in (3, 5):
        summary = rank_summary(ranks, list(range(2024, 2024 + n)))
        expected_rank = pd.read_csv(frozen / f"rank_reference_{n}seed.csv")
        merged = summary.merge(expected_rank, on=["horizon", "rank"], suffixes=("", "_reference"), validate="one_to_one")
        if len(merged) != len(summary) or len(merged) != len(expected_rank):
            raise ValueError("Rank reference grid mismatch")
        for c in ("macro_rmse_ns", "sample_sd_ns"):
            if not np.allclose(merged[c], merged[c + "_reference"], rtol=0, atol=1e-12):
                raise ValueError("Rank reference values mismatch")
        output.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output / f"rank_{n}seed.csv", index=False)
    spectra = pd.read_csv(frozen / "spectra.csv")
    if len(spectra) != 560 or spectra.duplicated(["horizon", "model", "seed", "satellite"]).any():
        raise ValueError("Unexpected spectral grid")
    for _, group in spectra.groupby(["horizon", "model", "seed"]):
        if set(group.satellite) != set(provenance["satellites"]):
            raise ValueError("Incomplete spectral satellite set")
    spectral_summary = spectra.groupby(["horizon", "model"])[["raw_r95", "context_r95", "task_r95"]].median().reset_index()
    v2 = spectral_summary[spectral_summary.model == "V2"].set_index("horizon")
    if not np.array_equal(v2.loc[["12h", "24h"], "raw_r95"], [2, 1.5]) or not (v2.task_r95 == 1).all():
        raise ValueError("Paper spectral medians mismatch")
    table.to_csv(output / "comparison.csv", index=False)
    display = table.copy()
    display["model"] = display["model"].replace(PAPER_NAMES)
    display.to_csv(output / "comparison_paper_names.csv", index=False)
    # Table III uses seed-averaged satellite scores, rather than pooled windows.
    satellite_means = records.groupby(["horizon", "model", "satellite"])["rmse_ns"].mean()
    distribution = satellite_means.groupby(["horizon", "model"]).agg(
        median_ns="median", p90_ns=lambda x: x.quantile(.90),
        p95_ns=lambda x: x.quantile(.95), worst_ns="max",
    ).reset_index()
    distribution["model"] = distribution["model"].replace(PAPER_NAMES)
    distribution.to_csv(output / "comparison_distribution.csv", index=False)
    seed_macro.to_csv(output / "comparison_seed_macro.csv", index=False)
    spectral_summary.to_csv(output / "spectral_medians.csv", index=False)
    report = {"passed": True, **integrity, "comparison_rows": len(records), "rank_rows": len(ranks), "spectral_rows": len(spectra), "max_reference_error_ns": max_error, "satellite_count": 28, "seeds": provenance["seeds"]}
    write_json(output / "verification.json", report)
    return report


def figures(repo, output):
    from .plotting import figures as render_figures
    return render_figures(repo, output)
