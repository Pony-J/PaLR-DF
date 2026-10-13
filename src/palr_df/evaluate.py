"""Float32 forecasting followed by float64 clock reconstruction and window metrics."""
from __future__ import annotations
import numpy as np
import torch


def window_metrics(prediction, target):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 2 or not len(prediction):
        raise ValueError("Expected matching nonempty [window, horizon] arrays")
    error = prediction - target
    if not np.isfinite(error).all():
        raise ValueError("Non-finite clock error")
    return {
        "rmse_ns": np.sqrt(np.mean(error ** 2, axis=1)),
        "re_ns": np.ptp(error, axis=1),
        "eae_ns": np.abs(error[:, -1]),
        "mae_ns": np.mean(np.abs(error), axis=1),
        "bias_ns": np.mean(error, axis=1),
    }


def evaluate(model, samples, device="cpu"):
    # The original evaluator used one full-window batch per satellite.
    model.to(device).eval()
    with torch.inference_mode():
        differences = model(torch.from_numpy(samples.x_difference_ns).to(device)).cpu().numpy().astype(np.float64)
    predicted_bias = samples.anchor_bias_ns[:, None] + np.cumsum(differences, axis=1)
    windows = window_metrics(predicted_bias, samples.target_bias_ns)
    return {k: float(v.mean()) for k, v in windows.items()}, windows, differences, predicted_bias

