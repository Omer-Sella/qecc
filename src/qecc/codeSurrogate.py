"""
First attempt at a decoder surrogate: predicts a BB code's combined failure-rate curve.
The idea is to incorporate this into the critic component of PPO, to improve sample efficiency.

Input: exponent bits [aX | aY | bX | bY] plus (l, m, k). Output: 5 logits, one
per canonical error-rate point; sigmoid gives the combined
(logical + decoder-failure) rate. 
The predicted reward can then be calculated using a separate collapse layer:
trapezoid(1 - curve, errorRange). 
Trained with Binomial NLL against the logged counts. 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from qecc.attentionArchitectures import (
    TOKEN_FEATURE_SIZE, AttentionPool, CodeEncoder, buildTokenFeatures,
)


class CodeCurvePredictor(nn.Module):
    def __init__(self, dModel=64, nHead=4, numLayers=2, dimFeedforward=128,
                 numberOfHarmonics=3, curvePoints=5):
        super().__init__()
        self.numberOfHarmonics = numberOfHarmonics
        featureSize = 1 + 4 + 2 * numberOfHarmonics + 1 + 4
        assert numberOfHarmonics != 3 or featureSize == TOKEN_FEATURE_SIZE
        self.encoder = CodeEncoder(featureSize, dModel, nHead, numLayers, dimFeedforward)
        self.pool = AttentionPool(dModel)
        self.curveHead = nn.Sequential(
            nn.Linear(dModel, dModel), nn.GELU(), nn.Linear(dModel, curvePoints))

    def forward(self, bits, l, m, k):
        tokens = buildTokenFeatures(bits, l, m, k, self.numberOfHarmonics)
        return self.curveHead(self.pool(self.encoder(tokens)))

    def predictCurve(self, bits, l, m, k):
        return torch.sigmoid(self.forward(bits, l, m, k))


def binomialCurveLoss(logits, counts, samples):
    """Mean Binomial NLL (up to the log-binomial-coefficient constant).

    n * BCE(p, c/n) = -(c log p + (n - c) log(1 - p)). samples may vary per
    record because duplicate codes are merged by summing trials.
    """
    targets = counts / samples.unsqueeze(-1)
    perPoint = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (perPoint * samples.unsqueeze(-1)).mean()


def rewardFromCurve(curve, errorRange):
    grid = torch.as_tensor(errorRange, dtype=curve.dtype, device=curve.device)
    return torch.trapezoid(1.0 - curve, grid, dim=-1)
