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
    bits, _counts, _samples, _k = toTensors(data)
    curves = []
    kPredictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, bits.shape[0], batchSize):
            end = start + batchSize
            curveLogits, kLogPrediction = model(bits[start:end], data.l, data.m)
            curves.append(torch.sigmoid(curveLogits))
            kPredictions.append(torch.expm1(kLogPrediction))
    curve = torch.cat(curves).numpy()
    kPredicted = torch.cat(kPredictions).numpy()
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
        "kMae": float(np.abs(kPredicted - data.k).mean()),
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


def appendReport(reportPath, title, dataDescription, metrics):
    """Append one metrics block to the markdown report; return the text written."""
    text = "\n".join([
        f"\n## {title}\n",
        f"- data: `{dataDescription}`",
        f"- Binomial NLL: **{metrics['nll']:.4f}** (noise floor {metrics['noiseFloor']:.4f})",
        f"- reward MAE: **{metrics['rewardMae']:.5f}**",
        f"- k MAE: **{metrics['kMae']:.2f}** logical qubits",
        f"- Spearman: **{metrics['spearman']:.3f}**, Kendall: {metrics['kendall']:.3f}",
        f"- top-k overlap (k={min(50, max(1, metrics['numberOfCodes'] // 10))}): "
        f"**{metrics['topK50']:.2f}**\n"])
    os.makedirs(os.path.dirname(reportPath) or ".", exist_ok=True)
    with open(reportPath, "a", encoding="utf-8") as fid:
        fid.write(text)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--l", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--report", default=None,
                        help="Defaults to surrogate-transfer-report.md next to the checkpoint")
    arguments = parser.parse_args()
    # Default: append to the report in the checkpoint's own run folder, so a
    # training run and its cross-size transfer evaluations share one directory.
    reportPath = arguments.report or os.path.join(
        os.path.dirname(arguments.checkpoint) or ".", "surrogate-transfer-report.md")

    model = loadCheckpoint(arguments.checkpoint)
    data = loadCodeEvaluations(arguments.data_root, arguments.l, arguments.m)
    metrics = evaluateOnData(model, data)
    title = (f"{arguments.checkpoint} on l={arguments.l}, m={arguments.m} "
             f"({metrics['numberOfCodes']} codes) — {datetime.date.today().isoformat()}")
    print(appendReport(reportPath, title, arguments.data_root, metrics))


if __name__ == "__main__":
    main()
