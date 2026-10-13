"""Gap-safe clock windows with the historical first-paper alignment."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class Samples:
    x_difference_ns: np.ndarray
    y_difference_ns: np.ndarray
    anchor_bias_ns: np.ndarray
    target_bias_ns: np.ndarray
    target_start: np.ndarray

    def __len__(self):
        return len(self.x_difference_ns)


class ClockData:
    """Biases are float64 nanoseconds; datetime labels are GPST, without conversion."""
    def __init__(self, path: str | Path):
        with np.load(path, allow_pickle=False) as z:
            self.epochs = z["epochs"].astype("datetime64[ns]")
            self.satellites = z["satellites"].tolist()
            self.bias_ns = z["bias_ns"].astype(np.float64)
            self.sampling_seconds = int(z["sampling_seconds"])
            if str(z["time_system"]) != "GPS":
                raise ValueError("Expected GPS time labels")
        if self.sampling_seconds != 30 or self.bias_ns.shape != (len(self.epochs), len(self.satellites)):
            raise ValueError("Invalid clock-array dimensions or sampling")
        if len(set(self.satellites)) != len(self.satellites):
            raise ValueError("Duplicate satellite identifiers")
        if np.isnat(self.epochs).any() or not np.isfinite(self.bias_ns).all():
            raise ValueError("Clock data contain missing or non-finite values")
        if len(self.epochs) < 2 or np.any(np.diff(self.epochs.astype(np.int64)) <= 0):
            raise ValueError("Clock epochs must be strictly increasing")

    def samples(self, config: dict, satellite: str, split: str) -> Samples:
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        if satellite not in config["satellites"] or satellite not in self.satellites:
            raise ValueError(f"Unknown satellite: {satellite}")
        length = int(config["input_length"])
        horizon = int(config["prediction_length"])
        stride = int(config["training_stride" if split == "train" else "validation_stride"])
        if min(length, horizon, stride) <= 0 or int(config["sampling_seconds"]) != 30:
            raise ValueError("Invalid sampling/window configuration")
        history_days = (length + 2879) // 2880
        column = self.satellites.index(satellite)
        parts = []
        for start, end in config[f"{split}_target_intervals"]:
            if date.fromisoformat(end) <= date.fromisoformat(start):
                raise ValueError("Empty target interval")
            raw_start = date.fromisoformat(start) - timedelta(days=history_days)
            mask = (self.epochs >= np.datetime64(raw_start)) & (self.epochs < np.datetime64(end))
            epochs = self.epochs[mask]
            raw = self.bias_ns[mask, column]
            expected = (date.fromisoformat(end) - raw_start).days * 2880
            if len(epochs) != expected or epochs[0] != np.datetime64(raw_start) or not np.all(np.diff(epochs) == np.timedelta64(30, "s")):
                raise ValueError(f"Missing/noncontiguous source interval: {raw_start} to {end}")
            differences = np.diff(raw)
            candidate = np.arange(len(differences) - length - horizon + 1)
            eligible = candidate[(epochs[candidate + length + 1] >= np.datetime64(start)) & (epochs[candidate + length + horizon] < np.datetime64(end))]
            if len(eligible) == 0:
                raise ValueError("No complete target windows")
            selected = eligible[(eligible - eligible[0]) % stride == 0]
            parts.append(Samples(
                np.stack([differences[i:i + length] for i in selected]).astype(np.float32),
                np.stack([differences[i + length:i + length + horizon] for i in selected]).astype(np.float32),
                raw[selected + length].copy(),
                np.stack([raw[i + length + 1:i + length + horizon + 1] for i in selected]),
                epochs[selected + length + 1].copy(),
            ))
        starts = np.concatenate([p.target_start for p in parts])
        if len(np.unique(starts)) != len(starts):
            raise ValueError("Overlapping target intervals create duplicate windows")
        order = np.argsort(starts)
        return Samples(*(np.concatenate([getattr(p, field) for p in parts])[order] for field in Samples.__dataclass_fields__))

