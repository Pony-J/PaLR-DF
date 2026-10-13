"""Context- and task-weighted spectra without dense horizon-square operators."""
from __future__ import annotations
import numpy as np
import torch


def energy_rank(singular, threshold=0.95):
    if not 0 < threshold <= 1:
        raise ValueError("Energy threshold must be in (0, 1]")
    square = np.sort(np.square(np.asarray(singular, dtype=np.float64)))[::-1]
    total = square.sum()
    if total <= np.finfo(np.float64).tiny:
        return 0
    return min(len(square), int(np.searchsorted(np.cumsum(square) / total, threshold, side="left") + 1))


def compact_singular_values(left, right):
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("Factor dimension mismatch")
    _, ra = np.linalg.qr(a, mode="reduced")
    _, rb = np.linalg.qr(b.T, mode="reduced")
    return np.sort(np.maximum(np.linalg.svd(ra @ rb.T, compute_uv=False), 0))[::-1]


def covariance(values):
    centered = np.asarray(values, dtype=np.float64) - np.mean(values, axis=0, keepdims=True)
    return centered.T @ centered / len(centered)


def psd_factor(matrix, symmetric=False):
    values, vectors = np.linalg.eigh(np.asarray(matrix, dtype=np.float64))
    floor = max(float(values.max()), np.finfo(np.float64).tiny) * 1e-12
    values = np.where(values > floor, values, 0)
    if symmetric:
        return (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.T
    return np.diag(np.sqrt(np.maximum(values, 0))) @ vectors.T


def loss_gram(basis_rows, difference_weight=1.0, clock_weight=1.0, endpoint_weight=0.1):
    left = np.asarray(basis_rows, dtype=np.float64).T
    horizon = len(left)
    integrated = np.cumsum(left, axis=0)
    endpoint = left.sum(axis=0)
    return difference_weight * (left.T @ left / horizon) + clock_weight * (integrated.T @ integrated / horizon) + endpoint_weight * np.outer(endpoint, endpoint)


def diagnose(model, samples, training_config, device="cpu", batch_size=64):
    model.to(device).eval()
    contexts = []
    handle = model.fusion.register_forward_hook(lambda _m, _i, out: contexts.append(out.detach().cpu()))
    try:
        with torch.inference_mode():
            for start in range(0, len(samples), batch_size):
                model(torch.from_numpy(samples.x_difference_ns[start:start + batch_size]).to(device))
    finally:
        handle.remove()
    context = torch.cat(contexts).numpy().astype(np.float64)
    root_cov = psd_factor(covariance(context), symmetric=True)
    basis = model.shape_basis.detach().cpu().numpy().astype(np.float64)
    basis -= basis.mean(axis=1, keepdims=True)
    coefficient = model.coefficient_head.weight.detach().cpu().numpy().astype(np.float64)
    t = training_config
    gram = loss_gram(basis, float(t.get("difference_loss_weight", 1)), float(t["clock_loss_weight"]), float(t.get("endpoint_loss_weight", 0)))
    singular = {
        "raw": compact_singular_values(basis.T, coefficient),
        "context": compact_singular_values(basis.T, coefficient @ root_cov),
        "task": compact_singular_values(psd_factor(gram), coefficient @ root_cov),
    }
    return {"training_context_count": len(context), **{f"{name}_r95": energy_rank(s) for name, s in singular.items()}, "singular_values": {k: v.tolist() for k, v in singular.items()}}

