"""Local direct-forecasting comparison adapters used in the first paper.

Informer, SSA-TCN-D and MGAMNet are local implementations; see docs/protocol.md.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

class DirectLSTMAttention(nn.Module):
    """Direct-horizon LSTM-attention baseline using the paper's 240-point crop."""

    def __init__(
        self,
        input_length: int,
        prediction_length: int,
        recent_length: int = 240,
        hidden_dim: int = 32,
        attention_heads: int = 1,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.recent_length = min(recent_length, input_length)
        self.input_projection = nn.Linear(1, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.attention = nn.MultiheadAttention(
            hidden_dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, prediction_length)
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim == 3:
            values = values.squeeze(-1)
        recent = values[:, -self.recent_length :]
        mean = recent.mean(dim=1, keepdim=True).detach()
        scale = recent.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5).detach()
        normalized = ((recent - mean) / scale).unsqueeze(-1)
        recurrent, _ = self.lstm(self.input_projection(normalized))
        attended, _ = self.attention(recurrent, recurrent, recurrent, need_weights=False)
        context = self.norm(recurrent + attended)[:, -1]
        return self.decoder(context) * scale + mean


class DirectLSTM(nn.Module):
    """Direct-horizon LSTM baseline using the paper's 240-epoch crop."""

    def __init__(
        self,
        input_length: int,
        prediction_length: int,
        recent_length: int = 240,
        hidden_dim: int = 48,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.recent_length = min(recent_length, input_length)
        self.input_projection = nn.Linear(1, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.decoder = nn.Linear(hidden_dim * 2, prediction_length)
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim == 3:
            values = values.squeeze(-1)
        recent = values[:, -self.recent_length :]
        mean = recent.mean(dim=1, keepdim=True).detach()
        scale = recent.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5).detach()
        normalized = ((recent - mean) / scale).unsqueeze(-1)
        recurrent, _ = self.lstm(self.input_projection(normalized))
        context = self.norm(torch.cat((recurrent.mean(dim=1), recurrent[:, -1]), dim=1))
        return self.decoder(context) * scale + mean


class DirectCNN(nn.Module):
    """Direct-horizon temporal CNN baseline using the paper's local crop."""

    def __init__(
        self,
        input_length: int,
        prediction_length: int,
        recent_length: int = 240,
        channels: int = 64,
        pooled_tokens: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.recent_length = min(recent_length, input_length)
        self.pooled_tokens = pooled_tokens
        self.encoder = nn.Sequential(
            nn.Conv1d(1, channels, 7, padding=3),
            nn.GELU(),
            nn.Conv1d(channels, channels, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.GELU(),
        )
        feature_count = channels * pooled_tokens * 2
        self.decoder = nn.Sequential(
            nn.Linear(feature_count, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, prediction_length),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim == 3:
            values = values.squeeze(-1)
        recent = values[:, -self.recent_length :]
        mean = recent.mean(dim=1, keepdim=True).detach()
        scale = recent.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5).detach()
        normalized = ((recent - mean) / scale).unsqueeze(1)
        features = self.encoder(normalized)
        summary = torch.cat(
            (
                F.adaptive_avg_pool1d(features, self.pooled_tokens),
                F.adaptive_max_pool1d(features, self.pooled_tokens),
            ),
            dim=1,
        ).flatten(1)
        return self.decoder(summary) * scale + mean


class CausalTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.norm = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = F.pad(values, (self.left_padding, 0))
        values = self.dropout(F.gelu(self.norm(self.pointwise(self.depthwise(values)))))
        return residual + values


class SSATCNApproxDirect(nn.Module):
    """Engineering approximation of SSA decomposition plus three TCN branches.

    The three additive bands are generated with fixed centered smoothers rather
    than an SVD-based SSA implementation, so results must retain the ``Approx.``
    label in paper tables.
    """

    def __init__(
        self,
        input_length: int,
        prediction_length: int,
        patch_size: int = 12,
        channels: int = 48,
        layers: int = 4,
        kernel_size: int = 3,
        pooled_tokens: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.pooled_tokens = pooled_tokens
        self.input_projections = nn.ModuleList([nn.Conv1d(1, channels, 1) for _ in range(3)])
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        CausalTCNBlock(channels, kernel_size, 2**level, dropout)
                        for level in range(layers)
                    ]
                )
                for _ in range(3)
            ]
        )
        feature_count = 3 * channels * pooled_tokens * 2
        self.decoder = nn.Sequential(
            nn.Linear(feature_count, 192),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(192, prediction_length),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    @staticmethod
    def _smooth(values: torch.Tensor, kernel_size: int) -> torch.Tensor:
        half = kernel_size // 2
        padded = F.pad(values, (half, half), mode="replicate")
        return F.avg_pool1d(padded, kernel_size, stride=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim == 3:
            values = values.squeeze(-1)
        recent_day = values[:, -min(2880, values.shape[1]) :]
        mean = recent_day.mean(dim=1, keepdim=True).detach()
        scale = recent_day.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5).detach()
        normalized = ((values - mean) / scale).unsqueeze(1)
        short = self._smooth(normalized, 13)
        low = self._smooth(normalized, 121)
        bands = (low, short - low, normalized - short)
        summaries = []
        for band, projection, branch in zip(bands, self.input_projections, self.branches):
            patched = F.avg_pool1d(band, self.patch_size, stride=self.patch_size)
            encoded = branch(projection(patched))
            summaries.extend(
                (
                    F.adaptive_avg_pool1d(encoded, self.pooled_tokens),
                    F.adaptive_max_pool1d(encoded, self.pooled_tokens),
                )
            )
        context = torch.cat(summaries, dim=1).flatten(1)
        return self.decoder(context) * scale + mean


class InformerDirect(nn.Module):
    """Standalone distilled-attention direct baseline for long inputs.

    A strided convolution tokenizes the 30 s series before two attention
    encoder layers.  This keeps the baseline practical on a 12 GB GPU while
    retaining Informer's encoder-distillation idea.  It is reported as a
    standalone approximation, not as byte-identical TS-Lib Informer code.
    """

    def __init__(
        self,
        input_length: int,
        prediction_length: int,
        patch_size: int = 12,
        d_model: int = 32,
        attention_heads: int = 4,
        layers: int = 2,
        feedforward_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        token_count = (input_length + patch_size - 1) // patch_size
        self.embedding = nn.Conv1d(
            1, d_model, kernel_size=patch_size, stride=patch_size
        )
        self.position = nn.Parameter(torch.zeros(1, token_count, d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=layers, norm=nn.LayerNorm(d_model)
        )
        self.decoder = nn.Linear(d_model * 2, prediction_length)
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim == 3:
            values = values.squeeze(-1)
        mean = values.mean(dim=1, keepdim=True).detach()
        scale = values.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5).detach()
        normalized = ((values - mean) / scale).unsqueeze(1)
        tokens = self.embedding(normalized).transpose(1, 2)
        tokens = tokens + self.position[:, : tokens.shape[1]]
        encoded = self.encoder(tokens)
        context = torch.cat((encoded.mean(dim=1), encoded[:, -1]), dim=1)
        return self.decoder(context) * scale + mean


class _MGAMMixBlock(nn.Module):
    """Cross-granularity mixing followed by scale-specific self-attention."""

    def __init__(
        self,
        granularity_count: int,
        d_model: int,
        attention_heads: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.mixers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(granularity_count * d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, d_model),
                )
                for _ in range(granularity_count)
            ]
        )
        self.attention = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=attention_heads,
                    dim_feedforward=feedforward_dim,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(granularity_count)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(granularity_count)]
        )

    def forward(self, scales: list[torch.Tensor]) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        for target_index, target in enumerate(scales):
            target_length = target.shape[1]
            aligned = []
            for source in scales:
                if source.shape[1] == target_length:
                    aligned.append(source)
                else:
                    resized = F.interpolate(
                        source.transpose(1, 2),
                        size=target_length,
                        mode="linear",
                        align_corners=False,
                    ).transpose(1, 2)
                    aligned.append(resized)
            mixed = self.mixers[target_index](torch.cat(aligned, dim=-1))
            mixed = self.norms[target_index](target + mixed)
            outputs.append(self.attention[target_index](mixed))
        return outputs


class MGAMNetApproxDirect(nn.Module):
    """MGAMNet-style direct baseline for the frozen per-satellite protocol.

    The implementation follows the publicly described MGP -> MGM -> MHSA
    hierarchy: several non-overlapping patch granularities construct parallel
    subsequences, every mixing block aligns and fuses all granularities in both
    directions, and scale-specific multi-head attention models dependencies.
    It must be reported as ``MGAMNet Approx.-D`` until the authors' full
    implementation or complete architecture specification is available.
    """

    def __init__(
        self,
        input_length: int,
        prediction_length: int,
        patch_sizes: tuple[int, ...] | list[int] = (12, 60, 120),
        d_model: int = 32,
        attention_heads: int = 4,
        layers: int = 2,
        feedforward_dim: int = 128,
        pooled_tokens: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        patches = tuple(int(value) for value in patch_sizes)
        if not patches or min(patches) <= 0 or prediction_length <= 0:
            raise ValueError("Invalid MGAMNet direct geometry")
        if len(set(patches)) != len(patches):
            raise ValueError("MGAMNet patch sizes must be unique")
        if max(patches) > input_length:
            raise ValueError("MGAMNet patch size exceeds input length")
        if d_model % attention_heads:
            raise ValueError("d_model must be divisible by attention_heads")
        if pooled_tokens <= 0:
            raise ValueError("pooled_tokens must be positive")
        self.patch_sizes = patches
        self.pooled_tokens = pooled_tokens
        self.embeddings = nn.ModuleList(
            [
                nn.Conv1d(1, d_model, kernel_size=patch_size, stride=patch_size)
                for patch_size in patches
            ]
        )
        self.positions = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(1, input_length // patch_size, d_model)
                )
                for patch_size in patches
            ]
        )
        for position in self.positions:
            nn.init.trunc_normal_(position, std=0.02)
        self.blocks = nn.ModuleList(
            [
                _MGAMMixBlock(
                    len(patches),
                    d_model,
                    attention_heads,
                    feedforward_dim,
                    dropout,
                )
                for _ in range(layers)
            ]
        )
        self.scale_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        summary_width = len(patches) * 2 * pooled_tokens * d_model
        self.decoder = nn.Sequential(
            nn.Linear(summary_width, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, prediction_length),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim == 3:
            values = values.squeeze(-1)
        recent = values[:, -min(2880, values.shape[1]) :]
        mean = recent.mean(dim=1, keepdim=True).detach()
        scale = recent.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5).detach()
        normalized = ((values - mean) / scale).unsqueeze(1)
        scales = []
        for embedding, position in zip(self.embeddings, self.positions):
            tokens = embedding(normalized).transpose(1, 2)
            scales.append(tokens + position[:, : tokens.shape[1]])
        for block in self.blocks:
            scales = block(scales)

        gate_logits = torch.cat(
            [self.scale_gate(tokens.mean(dim=1)) for tokens in scales], dim=1
        )
        gate_weights = torch.softmax(gate_logits, dim=1)
        summaries = []
        for index, tokens in enumerate(scales):
            channel_first = tokens.transpose(1, 2)
            pooled = torch.cat(
                (
                    F.adaptive_avg_pool1d(channel_first, self.pooled_tokens),
                    F.adaptive_max_pool1d(channel_first, self.pooled_tokens),
                ),
                dim=2,
            )
            summaries.append(pooled.flatten(1) * gate_weights[:, index : index + 1])
        return self.decoder(torch.cat(summaries, dim=1)) * scale + mean

ADAPTERS = {"cnn_direct": DirectCNN, "lstm_direct": DirectLSTM, "lstm_attention_direct": DirectLSTMAttention, "informer_direct_approx": InformerDirect, "ssa_tcn_direct_approx": SSATCNApproxDirect, "mgamnet_direct_approx": MGAMNetApproxDirect}

