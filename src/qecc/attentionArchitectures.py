"""Size-transferable attention building blocks for BB-code models.

Every parameter shape depends only on (featureSize, dModel, nHead, numLayers,
dimFeedforward) - never on l, m, or the token count - so a state_dict trained
at one code size loads strict=True at any other. Positional information is
functional (cyclic sin/cos of the slot angle 2*pi*i/period), never learned
per slot. See docs/superpowers/specs/2026-07-11-code-evaluation-surrogate-design.md.
"""
import math

import torch
import torch.nn as nn

TOKEN_FEATURE_SIZE = 16          # 1 bit + 4 group + 2*3 harmonics + 1 linear + 4 globals
K_MIN_REFERENCE = 6.0


def buildTokenFeatures(bits, l, m, k, numberOfHarmonics=3):
    """bits: (B, 2l+2m) float, order [aX | aY | bX | bY]; k: (B,) float.

    Returns (B, 2l+2m, 1 + 4 + 2*numberOfHarmonics + 1 + 4) token features.
    """
    batchSize = bits.shape[0]
    device = bits.device
    groupPeriods = (l, m, l, m)  # aX, aY, bX, bY
    globalsPart = torch.stack([
        torch.full((batchSize,), math.log(l), device=device),
        torch.full((batchSize,), math.log(m), device=device),
        (k.to(torch.float32) / K_MIN_REFERENCE).clamp(0.0, 2.0),
        torch.ones(batchSize, device=device),
    ], dim=-1)                                                    # (B, 4)
    groupFeatures = []
    start = 0
    for groupIndex, period in enumerate(groupPeriods):
        positions = torch.arange(period, device=device, dtype=torch.float32)
        angle = 2.0 * math.pi * positions / period
        harmonicColumns = []
        for h in range(1, numberOfHarmonics + 1):
            harmonicColumns.append(torch.sin(h * angle))
            harmonicColumns.append(torch.cos(h * angle))
        positional = torch.stack(harmonicColumns + [positions / period], dim=-1)
        groupOneHot = torch.zeros(period, 4, device=device)
        groupOneHot[:, groupIndex] = 1.0
        static = torch.cat([groupOneHot, positional], dim=-1)     # (period, 4+2H+1)
        groupFeatures.append(torch.cat([
            bits[:, start:start + period].to(torch.float32).unsqueeze(-1),
            static.unsqueeze(0).expand(batchSize, -1, -1),
            globalsPart.unsqueeze(1).expand(-1, period, -1),
        ], dim=-1))
        start += period
    return torch.cat(groupFeatures, dim=1)


class CodeEncoder(nn.Module):
    def __init__(self, featureSize=TOKEN_FEATURE_SIZE, dModel=64, nHead=4,
                 numLayers=2, dimFeedforward=128):
        super().__init__()
        self.inputProjection = nn.Linear(featureSize, dModel)
        encoderLayer = nn.TransformerEncoderLayer(
            d_model=dModel, nhead=nHead, dim_feedforward=dimFeedforward,
            activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)

    def forward(self, tokens):
        return self.encoder(self.inputProjection(tokens))


class AttentionPool(nn.Module):
    """One learned query attends over all token embeddings -> fixed-size summary."""
    def __init__(self, dModel):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dModel) / math.sqrt(dModel))
        self.scale = math.sqrt(dModel)

    def forward(self, tokens):
        scores = tokens @ self.query / self.scale     # (B, N)
        weights = scores.softmax(dim=-1)
        return (weights.unsqueeze(-1) * tokens).sum(dim=1)
