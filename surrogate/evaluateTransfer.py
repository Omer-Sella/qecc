# surrogate/evaluateTransfer.py
"""Evaluate a trained surrogate on a labelled dataset; report transfer metrics.

Metrics: Binomial NLL vs. the empirical noise floor, reward MAE, Spearman and
Kendall rank correlation of predicted vs. true reward, top-50 retrieval overlap.

Usage:
    python surrogate/evaluateTransfer.py --checkpoint surrogate/checkpoints/surrogate_6x6.pth \
        --data-root C:/Users/Omer/rl-qecc-data/probes_9x9 --l 9 --m 9 \
        --report docs/surrogate/transfer-report.md
"""
import argparse
import datetime
import os

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr

from qecc.codeEvaluationDataset import (CANONICAL_ERROR_RANGE, loadCodeEvaluations,
                                        rewardFromCounts, toTensors)
from qecc.codeSurrogate import CodeCurvePredictor, rewardFromCurve

EPSILON = 1e-6


def binomialNllOfCurve(curve, counts, samples):
    p = np.clip(curve, EPSILON, 1.0 - EPSILON)
    n = samples[:, None]
    nll = -(counts * np.log(p) + (n - counts) * np.log(1.0 - p))
    return float(nll.mean())


def noiseFloorNll(counts, samples):
    return binomialNllOfCurve(counts / samples[:, None], counts, samples)


def topKOverlap(trueValues, predictedValues, k):
    trueTop = set(np.argsort(trueValues)[-k:])
    predictedTop = set(np.argsort(predictedValues)[-k:])
    return len(trueTop & predictedTop) / k


def evaluateOnData(model, data, batchSize=1024):
    bits, _counts, _samples, k = toTensors(data)
    curves = []
    model.eval()
    with torch.no_grad():
        for start in range(0, bits.shape[0], batchSize):
            end = start + batchSize
            curves.append(model.predictCurve(bits[start:end], data.l, data.m,
                                             k[start:end]))
    curve = torch.cat(curves).numpy()
    trueReward = rewardFromCounts(data.counts, data.samples, CANONICAL_ERROR_RANGE)
    predictedReward = rewardFromCurve(torch.as_tensor(curve),
                                      CANONICAL_ERROR_RANGE).numpy()
    kTop = min(50, max(1, data.bits.shape[0] // 10))
    # Rank correlation is genuinely undefined when either series is constant
    # (zero range); short-circuit instead of calling scipy, which would emit
    # a RuntimeWarning and still return NaN.
    if np.ptp(trueReward) == 0 or np.ptp(predictedReward) == 0:
        spearman = float("nan")
        kendall = float("nan")
    else:
        spearman = float(spearmanr(trueReward, predictedReward).statistic)
        kendall = float(kendalltau(trueReward, predictedReward).statistic)
    return {
        "nll": binomialNllOfCurve(curve, data.counts.astype(float),
                                  data.samples.astype(float)),
        "noiseFloor": noiseFloorNll(data.counts.astype(float),
                                    data.samples.astype(float)),
        "rewardMae": float(np.abs(trueReward - predictedReward).mean()),
        "spearman": spearman,
        "kendall": kendall,
        "topK50": topKOverlap(trueReward, predictedReward, kTop),
        "numberOfCodes": int(data.bits.shape[0]),
    }


def loadCheckpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = CodeCurvePredictor(**checkpoint["hyperParameters"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--l", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--report", default="docs/surrogate/transfer-report.md")
    arguments = parser.parse_args()

    model = loadCheckpoint(arguments.checkpoint)
    data = loadCodeEvaluations(arguments.data_root, arguments.l, arguments.m)
    metrics = evaluateOnData(model, data)

    lines = [f"\n## {arguments.checkpoint} on l={arguments.l}, m={arguments.m} "
             f"({metrics['numberOfCodes']} codes) — "
             f"{datetime.date.today().isoformat()}\n",
             f"- data: `{arguments.data_root}`",
             f"- Binomial NLL: **{metrics['nll']:.4f}** (noise floor {metrics['noiseFloor']:.4f})",
             f"- reward MAE: **{metrics['rewardMae']:.5f}**",
             f"- Spearman: **{metrics['spearman']:.3f}**, Kendall: {metrics['kendall']:.3f}",
             f"- top-k overlap (k={min(50, max(1, metrics['numberOfCodes'] // 10))}): "
             f"**{metrics['topK50']:.2f}**\n"]
    os.makedirs(os.path.dirname(arguments.report) or ".", exist_ok=True)
    with open(arguments.report, "a") as fid:
        fid.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
