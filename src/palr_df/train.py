"""The paper's Adam recipe, with explicit output and provenance."""
from __future__ import annotations
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .evaluate import evaluate


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def initialize_pca(model, samples):
    recent = samples.x_difference_ns[:, -2880:].astype(np.float64)
    normalized = (samples.y_difference_ns.astype(np.float64) - recent.mean(axis=1, keepdims=True)) / np.maximum(recent.std(axis=1, keepdims=True), 1e-5)
    normalized -= normalized.mean(axis=1, keepdims=True)
    _, _, right = np.linalg.svd(normalized, full_matrices=False)
    rank = len(model.shape_basis)
    if len(right) < rank:
        raise ValueError("Insufficient training windows for PCA basis initialization")
    basis = right[:rank]
    basis /= np.sqrt(np.mean(basis ** 2, axis=1, keepdims=True)).clip(1e-12)
    with torch.no_grad():
        model.shape_basis.copy_(torch.from_numpy(basis).to(model.shape_basis))


def fit(model, train, validation, config, device="cpu"):
    t = config["training"]
    dataset = TensorDataset(torch.from_numpy(train.x_difference_ns), torch.from_numpy(train.y_difference_ns))
    loader = DataLoader(dataset, batch_size=min(int(t["batch_size"]), len(dataset)), shuffle=True, num_workers=0, pin_memory=str(device).startswith("cuda"))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(t["learning_rate"]))
    best = evaluate(model, validation, device)[0]["rmse_ns"]
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch, stale, history = 0, 0, []
    for epoch in range(1, int(t["max_epochs"]) + 1):
        model.train()
        total, seen = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            residual = prediction - y
            loss = float(t.get("difference_loss_weight", 1)) * nn.functional.mse_loss(prediction, y) + float(t["clock_loss_weight"]) * torch.mean(torch.cumsum(residual, dim=1) ** 2) + float(t.get("endpoint_loss_weight", 0)) * torch.mean(torch.sum(residual, dim=1) ** 2)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(t["gradient_clip_norm"]))
            optimizer.step()
            total += float(loss.detach()) * len(x)
            seen += len(x)
        score = evaluate(model, validation, device)[0]["rmse_ns"]
        history.append({"epoch": epoch, "train_loss": total / seen, "validation_rmse_ns": score})
        print(f"epoch={epoch} train_loss={total / seen:.8g} validation_rmse_ns={score:.8g}", flush=True)
        if score < best - 1e-8:
            best, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= int(t["early_stopping_patience"]):
                break
    model.load_state_dict(best_state, strict=True)
    return best_epoch, history
