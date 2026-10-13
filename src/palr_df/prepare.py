"""Convert the manifest-listed CODE RINEX CLK products into the portable data asset."""
from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from .io import read_json, safe_path, sha256


def prepare_data(raw_dir, product_manifest, satellites, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifest = read_json(product_manifest)
    epochs, arrays = [], []
    for entry in manifest["rows"]:
        source = safe_path(raw_dir, entry["filename"])
        if sha256(source) != entry["sha256"]:
            raise ValueError(f"Raw product hash mismatch: {source.name}")
        rows = {sat: [] for sat in satellites}
        gps_header = False
        with source.open(encoding="ascii", errors="strict") as f:
            for line in f:
                if "TIME SYSTEM ID" in line and "GPS" in line[:10]:
                    gps_header = True
                fields = line.split()
                if len(fields) < 10 or fields[0] != "AS" or fields[1] not in rows:
                    continue
                stamp = datetime(*(int(v) for v in fields[2:7])) + timedelta(seconds=float(fields[7]))
                bias = float(fields[9].replace("D", "E")) * 1e9
                rows[fields[1]].append((stamp, bias))
        if not gps_header:
            raise ValueError(f"Missing GPS time-system header: {source.name}")
        day = np.datetime64(entry["date"], "ns")
        expected = day + np.arange(2880) * np.timedelta64(30, "s")
        matrix = []
        for satellite in satellites:
            ordered = sorted(rows[satellite])
            times = np.asarray([r[0] for r in ordered], dtype="datetime64[ns]")
            values = np.asarray([r[1] for r in ordered], dtype=np.float64)
            if not np.array_equal(times, expected) or not np.isfinite(values).all():
                raise ValueError(f"Incomplete/duplicate clock day: {source.name} {satellite}")
            matrix.append(values)
        epochs.append(expected)
        arrays.append(np.stack(matrix, axis=1))
    output.parent.mkdir(parents=True, exist_ok=True)
    # Write through an open handle so numpy cannot silently append a suffix.
    with output.open("xb") as f:
        np.savez_compressed(f, epochs=np.concatenate(epochs), satellites=np.asarray(satellites), bias_ns=np.concatenate(arrays), time_system=np.asarray("GPS"), sampling_seconds=np.asarray(30))
    return {"raw_days": len(epochs), "satellites": len(satellites), "epochs": sum(map(len, epochs)), "sha256": sha256(output)}

