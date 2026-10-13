"""Persistence-augmented low-rank direct forecasting.

Forward operations and state-dict keys preserve the archived paper model.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

class MultiScaleConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernels: tuple[int, ...]):
        super().__init__()
        branch_channels = max(1, out_channels // len(kernels))
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(in_channels, branch_channels, kernel, padding=kernel // 2),
                    nn.GELU(),
                    nn.Conv1d(
                        branch_channels,
                        branch_channels,
                        3,
                        padding=1,
                        groups=branch_channels,
                    ),
                    nn.GELU(),
                )
                for kernel in kernels
            ]
        )
        merged_channels = branch_channels * len(kernels)
        self.projection = nn.Conv1d(merged_channels, out_channels, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.projection(torch.cat([branch(values) for branch in self.branches], dim=1))


class Periodic2DEncoder(nn.Module):
    """Encode several candidate periods after 2-D temporal reshaping."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        periods: tuple[int, ...] = (12, 24, 48, 60),
    ):
        super().__init__()
        self.periods = periods
        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(
                        out_channels,
                        out_channels,
                        3,
                        padding=1,
                        groups=out_channels,
                    ),
                    nn.GELU(),
                )
                for _ in periods
            ]
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, channels, length = values.shape
        spectrum = torch.fft.rfft(values.mean(dim=1), dim=-1).abs()
        representations = []
        amplitudes = []
        for period, encoder in zip(self.periods, self.encoders):
            padded_length = ((length + period - 1) // period) * period
            if padded_length != length:
                reshaped_input = F.pad(values, (0, padded_length - length))
            else:
                reshaped_input = values
            grid = reshaped_input.reshape(batch, channels, padded_length // period, period)
            encoded = encoder(grid)
            average = F.adaptive_avg_pool2d(encoded, 1).flatten(1)
            maximum = F.adaptive_max_pool2d(encoded, 1).flatten(1)
            representations.append(torch.cat((average, maximum), dim=1))
            frequency_index = min(max(1, length // period), spectrum.shape[-1] - 1)
            amplitudes.append(spectrum[:, frequency_index])
        weights = torch.softmax(torch.stack(amplitudes, dim=1), dim=1)
        stacked = torch.stack(representations, dim=1)
        return torch.sum(stacked * weights.unsqueeze(-1), dim=1)


class PaLRDF(nn.Module):
    """Slow/fast encoding with persistence and a centered learned residual basis.

    Inputs and outputs are clock differences in nanoseconds, with shapes
    [batch, input_length] and [batch, prediction_length], respectively.
    The defaults are the compact rank-3 paper model. Explicit wider settings
    load the V2 reference checkpoints through the same state-dict interface.
    """

    def __init__(
        self,
        input_length: int,
        prediction_length: int,
        slow_pool: int = 24,
        fast_length: int = 240,
        channels: int = 32,
        fusion_dim: int = 64,
        attention_heads: int = 4,
        pooled_tokens: int = 12,
        basis_count: int = 3,
        dropout: float = 0.1,
        use_trend_anchor: bool = False,
        robust_normalization: bool = False,
        normalization_hours: int = 6,
        robust_clip: float = 12.0,
    ):
        super().__init__()
        if min(prediction_length, slow_pool, fast_length, channels, attention_heads, pooled_tokens, basis_count, fusion_dim) <= 0:
            raise ValueError("Model dimensions must be positive")
        if input_length % slow_pool:
            raise ValueError("input_length must be divisible by slow_pool")
        if input_length < 2880:
            raise ValueError("PaLRDF requires at least one 24 h input cycle")
        if channels % attention_heads != 0:
            raise ValueError("channels must be divisible by attention_heads")
        self.input_length = input_length
        self.prediction_length = prediction_length
        self.slow_pool = slow_pool
        self.fast_length = min(fast_length, input_length)
        self.pooled_tokens = pooled_tokens
        self.use_trend_anchor = use_trend_anchor
        self.robust_normalization = robust_normalization
        self.normalization_hours = normalization_hours
        self.robust_clip = robust_clip
        if normalization_hours < 1 or normalization_hours * 120 > input_length:
            raise ValueError("normalization_hours must fit the input")

        self.slow_encoder = MultiScaleConv1d(3, channels, (3, 5, 9))
        self.periodic_encoder = Periodic2DEncoder(3, channels // 2)
        self.fast_encoder = MultiScaleConv1d(1, channels, (1, 3, 5, 7))
        self.cross_attention = nn.MultiheadAttention(
            channels, attention_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(channels)
        self.cross_gate = nn.Linear(channels * 2, channels)

        token_features = channels * 3
        periodic_features = channels
        statistical_features = 24 + 4 + 4
        fused_features = token_features + periodic_features + statistical_features
        self.fusion = nn.Sequential(
            nn.Linear(fused_features, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
        )
        self.mean_head = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.scale_head = nn.Linear(fusion_dim, 1)
        self.persistence_head = nn.Linear(fusion_dim, 1)
        self.coefficient_head = nn.Linear(fusion_dim, basis_count)
        self.shape_basis = nn.Parameter(
            torch.randn(basis_count, prediction_length) * 0.01
        )
        self.trend_logit = nn.Parameter(torch.tensor(0.0))
        nn.init.zeros_(self.mean_head[-1].weight)
        nn.init.zeros_(self.mean_head[-1].bias)
        nn.init.zeros_(self.scale_head.weight)
        nn.init.zeros_(self.scale_head.bias)
        nn.init.zeros_(self.persistence_head.weight)
        nn.init.constant_(self.persistence_head.bias, -2.0)
        nn.init.zeros_(self.coefficient_head.weight)
        nn.init.zeros_(self.coefficient_head.bias)

    @staticmethod
    def _decompose(values: torch.Tensor) -> torch.Tensor:
        short = F.avg_pool1d(values, 5, stride=1, padding=2)
        low = F.avg_pool1d(values, 25, stride=1, padding=12)
        mid = short - low
        high = values - short
        return torch.cat((low, mid, high), dim=1)

    @staticmethod
    def _hourly_statistics(
        values: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        recent_day = values[:, -2880:]
        hourly = recent_day.reshape(recent_day.shape[0], 24, 120).mean(dim=-1)
        normalized_hourly = (hourly - mean) / scale
        quarter = normalized_hourly.reshape(normalized_hourly.shape[0], 4, 6).mean(-1)
        within_day_slope = normalized_hourly[:, -1:] - normalized_hourly[:, :1]
        if values.shape[1] >= 5760:
            previous_mean = values[:, -5760:-2880].mean(dim=1, keepdim=True)
            day_drift = (mean - previous_mean) / scale
        else:
            day_drift = torch.zeros_like(mean)
        physical = torch.cat(
            (mean, torch.log(scale), within_day_slope, day_drift), dim=1
        )
        return torch.cat((normalized_hourly, quarter, physical), dim=1)

    def _persistence_anchor(self, normalized: torch.Tensor) -> torch.Tensor:
        recent_day = normalized[:, -2880:]
        if self.prediction_length <= 2880:
            anchor = recent_day[:, : self.prediction_length]
        else:
            repeats = (self.prediction_length + 2879) // 2880
            anchor = recent_day.repeat(1, repeats)[:, : self.prediction_length]
        return anchor - anchor.mean(dim=1, keepdim=True)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim == 3:
            values = values.squeeze(-1)
        if values.ndim != 2 or values.shape[1] != self.input_length:
            raise ValueError(f"Expected [batch, {self.input_length}] clock differences")
        if self.robust_normalization:
            normalization_window = values[:, -self.normalization_hours * 120 :]
            mean = normalization_window.median(dim=1, keepdim=True).values.detach()
            recent_day = values[:, -2880:]
            day_median = recent_day.median(dim=1, keepdim=True).values
            mad = (recent_day - day_median).abs().median(dim=1, keepdim=True).values
            scale = (1.4826 * mad).clamp_min(1e-5).detach()
        else:
            mean = values[:, -2880:].mean(dim=1, keepdim=True).detach()
            scale = (
                values[:, -2880:]
                .std(dim=1, keepdim=True, unbiased=False)
                .clamp_min(1e-5)
                .detach()
            )
        normalized = (values - mean) / scale
        if self.robust_normalization:
            normalized = torch.clamp(normalized, -self.robust_clip, self.robust_clip)

        slow = F.avg_pool1d(
            normalized.unsqueeze(1), self.slow_pool, stride=self.slow_pool
        )
        slow_bands = self._decompose(slow)
        slow_features = self.slow_encoder(slow_bands)
        periodic = self.periodic_encoder(slow_bands)

        fast = normalized[:, -self.fast_length :].unsqueeze(1)
        fast_features = self.fast_encoder(fast)
        slow_tokens = F.adaptive_avg_pool1d(slow_features, self.pooled_tokens).transpose(1, 2)
        fast_tokens = F.adaptive_avg_pool1d(fast_features, self.pooled_tokens).transpose(1, 2)
        attended, _ = self.cross_attention(
            fast_tokens, slow_tokens, slow_tokens, need_weights=False
        )
        attended = self.cross_norm(fast_tokens + attended)
        gate = torch.sigmoid(self.cross_gate(torch.cat((fast_tokens, attended), dim=-1)))
        fused_tokens = gate * fast_tokens + (1.0 - gate) * attended
        token_summary = torch.cat(
            (
                fused_tokens.mean(dim=1),
                fused_tokens.amax(dim=1),
                fused_tokens[:, -1],
            ),
            dim=1,
        )
        statistics = self._hourly_statistics(values, mean, scale)
        if self.robust_normalization:
            statistics = torch.clamp(statistics, -self.robust_clip, self.robust_clip)
        context = self.fusion(torch.cat((token_summary, periodic, statistics), dim=1))

        mean_adjustment = torch.tanh(self.mean_head(context)) * scale
        if values.shape[1] >= 5760:
            previous_values = values[:, -5760:-2880]
            previous_mean = (
                previous_values.median(dim=1, keepdim=True).values
                if self.robust_normalization
                else previous_values.mean(dim=1, keepdim=True)
            )
            day_delta = mean - previous_mean
        else:
            day_delta = torch.zeros_like(mean)
        trend_weight = torch.sigmoid(self.trend_logit) if self.use_trend_anchor else 0.0
        horizon_fraction = min(1.0, self.prediction_length / 2880.0)
        future_mean = mean + trend_weight * horizon_fraction * day_delta + mean_adjustment
        scale_ratio = torch.exp(torch.clamp(self.scale_head(context), -0.7, 0.7))
        persistence_gate = torch.sigmoid(self.persistence_head(context))
        residual_shape = self.coefficient_head(context) @ self.shape_basis
        residual_shape = residual_shape - residual_shape.mean(dim=1, keepdim=True)
        persistence = self._persistence_anchor(normalized)
        future_shape = persistence_gate * persistence + residual_shape
        future_shape = future_shape - future_shape.mean(dim=1, keepdim=True)
        return future_mean + future_shape * scale * scale_ratio

def from_config(config: dict) -> nn.Module:
    options = dict(config["model"])
    name = options.pop("name")
    if name not in {"palr", "v2"}:
        from .baselines import ADAPTERS
        if name not in ADAPTERS:
            raise ValueError(f"Unsupported model family: {name}")
        return ADAPTERS[name](int(config["input_length"]), int(config["prediction_length"]), **options)
    return PaLRDF(int(config["input_length"]), int(config["prediction_length"]), **options)
