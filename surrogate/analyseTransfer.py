# surrogate/analyseTransfer.py
"""Extended, decision-oriented analysis of surrogate transfer results.

Builds on evaluateTransfer.py (which reports point metrics) and adds the analyses that
make those numbers interpretable:

- bootstrap confidence intervals on every metric (resampling codes),
- PAIRED model-vs-model comparison on shared resamples (much tighter than separate CIs),
- regret@k: how much true reward is lost by trusting the surrogate's top-k picks,
- k-regime stratification: the same metrics restricted to codes with k >= kMin,
- a label-noise ceiling: the best Spearman ANY model could reach against finite-sample
  labels (replicate reliability, and its sqrt as the perfect-model ceiling),
- a calibration (reliability) table with expected calibration error, and an affine
  recalibration check (is bad MAE just a fixable scale/offset?).

Appends a markdown block per invocation and one CSV row per checkpoint, so running the
script across sizes accumulates the data for the metric-vs-period-distance figure.

Usage (all sizes in one go; per-size l_{l}_m_{m}/ subfolders are found automatically):
    python surrogate/analyseTransfer.py \
        --checkpoints A/surrogate_6x6.pth B/surrogate_6x6.pth C/surrogate_6x6.pth \
        --data-root $QECC_DATA/supervisedLearning/codeEvaluationTrainingData \
        --sizes 9,6 12,6 15,3 21,18 --k-min 8

Single size (equivalent alternative): --l 9 --m 6 with --data-root pointing anywhere.
"""
import argparse
import csv
import datetime
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr
from qecc.utils import calculateRewardFromSamples

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluateTransfer import loadCheckpoint, binomialNllOfCurve, noiseFloorNll  # noqa: E402

from qecc.codeEvaluationDataset import (loadCodeEvaluations,  # noqa: E402
                                        toTensors)
from qecc.utils import NAMED_ERROR_RANGES #, CANONICAL_ERROR_RANGE



# ---------------------------------------------------------------------------
# Per-code predictions
# ---------------------------------------------------------------------------

def predictPerCode(model, data, errorRange, batchSize=1024, rewardEngineering = True):
    """Forward the model over the whole dataset; return per-code predictions."""
    bits, _counts, _samples, _k = toTensors(data)
    curves, kPredictions = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, bits.shape[0], batchSize):
            curveLogits, kLogPrediction = model(bits[start:start + batchSize], data.l, data.m)
            curves.append(torch.sigmoid(curveLogits))
            kPredictions.append(torch.expm1(kLogPrediction))
    curve = torch.cat(curves).numpy()
    return {
        "curve": curve,
        "predictedReward": calculateRewardFromSamples(curve, numberOfSamples=1, #numberOfSamples=1 because the sigmoid curve is already a rate 
                                                      errorRange=errorRange,
                                                      l = data.l,
                                                      m = data.m,
                                                      rewardEngineering=rewardEngineering),
        "kPredicted": torch.cat(kPredictions).numpy(),
    }


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------

def spearmanSafe(a, b):
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def regretAtK(trueReward, predictedReward, k):
    """True reward foregone by taking the surrogate's top-k instead of the true best."""
    if len(trueReward) == 0:
        return float("nan")
    k = min(k, len(trueReward))
    topByPrediction = np.argsort(predictedReward)[-k:]
    return float(trueReward.max() - trueReward[topByPrediction].max())


def affineRecalibration(predictedReward, trueReward):
    """MAE before and after the best per-size affine correction a + b * prediction."""
    maeBefore = float(np.abs(trueReward - predictedReward).mean())
    if np.ptp(predictedReward) == 0:
        return maeBefore, maeBefore, 0.0, float(trueReward.mean())
    slope, intercept = np.polyfit(predictedReward, trueReward, 1)
    corrected = intercept + slope * predictedReward
    maeAfter = float(np.abs(trueReward - corrected).mean())
    return maeBefore, maeAfter, float(slope), float(intercept)


def calibrationTable(curve, counts, samples, numberOfBins=10):
    """Reliability table over all (code, grid point) cells, weighted by sample count.

    Returns (rows, ece): rows of (meanPredicted, empirical, weightFraction) per bin,
    binned by predicted-probability quantiles; ece is the weighted |gap|.
    """
    predicted = curve.flatten()
    observedCounts = counts.flatten().astype(float)
    cellSamples = np.repeat(samples.astype(float), curve.shape[1])
    edges = np.quantile(predicted, np.linspace(0, 1, numberOfBins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    rows = []
    ece = 0.0
    totalWeight = cellSamples.sum()
    for binIndex in range(numberOfBins):
        inBin = (predicted >= edges[binIndex]) & (predicted < edges[binIndex + 1])
        weight = cellSamples[inBin].sum()
        if weight == 0:
            continue
        meanPredicted = float(np.average(predicted[inBin], weights=cellSamples[inBin]))
        empirical = float(observedCounts[inBin].sum() / weight)
        rows.append((meanPredicted, empirical, weight / totalWeight))
        ece += (weight / totalWeight) * abs(meanPredicted - empirical)
    return rows, float(ece)


def labelNoiseCeiling(counts, samples, errorRange, l, m, numberOfReplicates=200, seed=0, rewardEngineering = True):
    """Replicate reliability of the labels themselves, and the implied model ceiling.

    Draw two synthetic replicates of every code's counts from Binomial(samples, pHat),
    correlate the two resulting reward vectors; repeat. Mean replicate-replicate Spearman
    = label reliability rho. Under classical attenuation (Pearson theory, used here as an
    approximation for Spearman), a PERFECT model's correlation against one noisy replicate
    is bounded by sqrt(rho).
    """
    rng = np.random.default_rng(seed)
    integerSamples = np.asarray(samples, dtype=np.int64)[:, None]   # binomial needs integer n
    pHat = np.clip(counts / integerSamples, 0.0, 1.0)
    grid = np.asarray(errorRange, dtype=float)
    correlations = np.empty(numberOfReplicates)
    for replicateIndex in range(numberOfReplicates):
        rewardA = calculateRewardFromSamples(rng.binomial(integerSamples, pHat), integerSamples, errorRange = errorRange, l = l, m = m, rewardEngineering = rewardEngineering)#np.trapezoid(1.0 - rng.binomial(integerSamples, pHat) / integerSamples,
                               #grid, axis=-1)
        rewardB = calculateRewardFromSamples(rng.binomial(integerSamples, pHat), integerSamples, errorRange = errorRange, l = l, m = m, rewardEngineering = rewardEngineering) # np.trapezoid(1.0 - rng.binomial(integerSamples, pHat) / integerSamples,
                               #grid, axis=-1)
        correlations[replicateIndex] = spearmanSafe(rewardA, rewardB)
    reliability = float(np.nanmean(correlations))
    ceiling = float(np.sqrt(max(reliability, 0.0)))
    low, high = np.nanpercentile(correlations, [2.5, 97.5])
    return reliability, (float(low), float(high)), ceiling


# ---------------------------------------------------------------------------
# Bootstrap machinery (shared resamples => paired comparisons)
# ---------------------------------------------------------------------------

def makeResampleIndices(numberOfCodes, numberOfResamples, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, numberOfCodes, size=(numberOfResamples, numberOfCodes))


def bootstrapStatistic(statisticFunction, resampleIndices):
    """Apply statisticFunction(indices) over shared resamples; return (point, low, high, samples)."""
    values = np.array([statisticFunction(indices) for indices in resampleIndices])
    low, high = np.nanpercentile(values, [2.5, 97.5])
    return float(low), float(high), values


def confidenceInterval(pointValue, statisticFunction, resampleIndices):
    low, high, values = bootstrapStatistic(statisticFunction, resampleIndices)
    return {"value": float(pointValue), "low": low, "high": high, "_samples": values}


# ---------------------------------------------------------------------------
# Per-model analysis
# ---------------------------------------------------------------------------

def analyseModel(predictions, trueReward, trueK, kMin, regretKs, resampleIndices):
    predictedReward = predictions["predictedReward"]
    result = {}

    result["spearman"] = confidenceInterval(
        spearmanSafe(trueReward, predictedReward),
        lambda idx: spearmanSafe(trueReward[idx], predictedReward[idx]),
        resampleIndices)

    stratum = np.where(trueK >= kMin)[0]
    result["stratumSize"] = int(len(stratum))
    if len(stratum) >= 3:
        result["spearmanStratum"] = confidenceInterval(
            spearmanSafe(trueReward[stratum], predictedReward[stratum]),
            lambda idx: spearmanSafe(trueReward[idx[np.isin(idx, stratum)]],
                                     predictedReward[idx[np.isin(idx, stratum)]]),
            resampleIndices)
    else:
        result["spearmanStratum"] = {"value": float("nan"), "low": float("nan"),
                                     "high": float("nan"), "_samples": None}

    result["regret"] = {}
    for k in regretKs:
        result["regret"][k] = confidenceInterval(
            regretAtK(trueReward, predictedReward, k),
            lambda idx, kk=k: regretAtK(trueReward[idx], predictedReward[idx], kk),
            resampleIndices)

    result["kMae"] = float(np.abs(predictions["kPredicted"] - trueK).mean())
    (result["maeBefore"], result["maeAfter"],
     result["affineSlope"], result["affineIntercept"]) = affineRecalibration(
        predictedReward, trueReward)
    return result


def pairedComparison(resultA, resultB):
    """CI on the Spearman DIFFERENCE (A - B) over the shared resamples."""
    samplesA = resultA["spearman"]["_samples"]
    samplesB = resultB["spearman"]["_samples"]
    difference = samplesA - samplesB
    low, high = np.nanpercentile(difference, [2.5, 97.5])
    probabilityABetter = float(np.nanmean(difference > 0))
    pointDifference = resultA["spearman"]["value"] - resultB["spearman"]["value"]
    return {"value": float(pointDifference), "low": float(low), "high": float(high),
            "pABetter": probabilityABetter}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def formatCi(entry, digits=3):
    return (f"**{entry['value']:.{digits}f}** "
            f"[{entry['low']:.{digits}f}, {entry['high']:.{digits}f}]")


def appendMarkdown(reportPath, text):
    os.makedirs(os.path.dirname(reportPath) or ".", exist_ok=True)
    with open(reportPath, "a", encoding="utf-8") as fid:
        fid.write(text)


def appendCsvRow(csvPath, row, fieldNames):
    newFile = not os.path.isfile(csvPath)
    os.makedirs(os.path.dirname(csvPath) or ".", exist_ok=True)
    with open(csvPath, "a", newline="", encoding="utf-8") as fid:
        writer = csv.DictWriter(fid, fieldnames=fieldNames)
        if newFile:
            writer.writeheader()
        writer.writerow(row)


CSV_FIELDS = ["date", "checkpoint", "dataRoot", "l", "m", "numberOfCodes", "kMin", "inSample",
              "spearman", "spearmanLow", "spearmanHigh",
              "spearmanStratum", "spearmanStratumLow", "spearmanStratumHigh", "stratumSize",
              "regret1", "regret1Low", "regret1High",
              "regret5", "regret5Low", "regret5High",
              "regret10", "regret10Low", "regret10High",
              "kMae", "maeBefore", "maeAfter", "affineSlope",
              "nll", "noiseFloor", "ece",
              "labelReliability", "labelCeiling"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="One or more surrogate checkpoints; >=2 enables paired comparison.")
    parser.add_argument("--error-range", default="linear5", choices=sorted(NAMED_ERROR_RANGES))
    parser.add_argument("--data-root", required=True,
                        help="Base data directory. With --sizes, a per-size subfolder "
                             "l_{l}_m_{m} is used when it exists (falls back to the base, "
                             "which works but walks every record).")
    parser.add_argument("--sizes", nargs="+", default=None, metavar="L,M",
                        help="One or more sizes as l,m pairs, e.g. --sizes 9,6 12,6 15,3 21,18. "
                             "Overrides --l/--m.")
    parser.add_argument("--l", type=int, default=None)
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--k-min", type=int, default=6,
                        help="Stratification threshold: metrics also reported for k >= this.")
    parser.add_argument("--regret-ks", default="1,5,10")
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="Number of bootstrap resamples.")
    parser.add_argument("--ceiling-replicates", type=int, default=200)
    parser.add_argument("--max-codes", type=int, default=0,
                        help="Random subsample cap for huge datasets (0 = use all). "
                             "The 6,6 pool has >1M codes; bootstrap on all of them is slow.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", default=None,
                        help="Markdown report path; default surrogate-transfer-analysis.md "
                             "next to the FIRST checkpoint.")
    parser.add_argument("--csv", default=None,
                        help="CSV path; default analysis.csv next to the report.")
    parser.add_argument("--reward-engineering", type = str, default="True", choices = ["true", "True", "false", "False"],
                            help="Whether or not we divide the integral by the width of the error range")
    arguments = parser.parse_args()

    reportPath = arguments.report or os.path.join(
        os.path.dirname(arguments.checkpoints[0]) or ".", "surrogate-transfer-analysis.md")
    csvPath = arguments.csv or os.path.join(os.path.dirname(reportPath) or ".", "analysis.csv")

    rewardEngineering = arguments.reward_engineering.lower() == "true"
    if arguments.sizes:
        sizes = []
        for token in arguments.sizes:
            l, m = (int(part) for part in token.split(","))
            sizes.append((l, m))
    elif arguments.l is not None and arguments.m is not None:
        sizes = [(arguments.l, arguments.m)]
    else:
        raise SystemExit("Specify sizes: either --sizes 9,6 12,6 ... or --l L --m M.")

    for l, m in sizes:
        sizeSubfolder = os.path.join(arguments.data_root, f"l_{l}_m_{m}")
        sizeDataRoot = sizeSubfolder if os.path.isdir(sizeSubfolder) else arguments.data_root
        print(f"\n===== analysing l={l}, m={m} from {sizeDataRoot} =====")
        try:
            analyseOneSize(arguments, l, m, sizeDataRoot, reportPath, csvPath, rewardEngineering=rewardEngineering)
        except ValueError as error:
            print(f"SKIP l={l}, m={m}: {error}")


def analyseOneSize(arguments, l, m, dataRoot, reportPath, csvPath, rewardEngineering):
    regretKs = [int(token) for token in arguments.regret_ks.split(",")]

    errorRange = NAMED_ERROR_RANGES[arguments.error_range]
    data = loadCodeEvaluations(dataRoot, l, m, errorRange=errorRange)
    rng = np.random.default_rng(arguments.seed)
    if arguments.max_codes and data.bits.shape[0] > arguments.max_codes:
        keep = rng.choice(data.bits.shape[0], size=arguments.max_codes, replace=False)
        from qecc.codeEvaluationDataset import _subset
        data = _subset(data, keep)
        subsampleNote = f" (subsampled to {arguments.max_codes})"
    else:
        subsampleNote = ""

    trueReward = calculateRewardFromSamples(data.counts, data.samples[:,None], 
                                            errorRange=errorRange, 
                                            l = data.l, m = data.m, rewardEngineering=rewardEngineering)
    trueK = data.k.astype(float)
    numberOfCodes = data.bits.shape[0]
    resampleIndices = makeResampleIndices(numberOfCodes, arguments.bootstrap, arguments.seed)

    reliability, reliabilityCi, ceiling = labelNoiseCeiling(
        data.counts.astype(float), data.samples.astype(float), errorRange, data.l, data.m, 
        arguments.ceiling_replicates, arguments.seed,  rewardEngineering)

    today = datetime.date.today().isoformat()
    lines = [f"\n# Analysis: l={l}, m={m} "
             f"({numberOfCodes} codes{subsampleNote}) — {today}\n",
             f"- data: `{dataRoot}`",
             f"- label replicate reliability (Spearman, label-vs-label): "
             f"**{reliability:.3f}** [{reliabilityCi[0]:.3f}, {reliabilityCi[1]:.3f}]",
             f"- implied PERFECT-model Spearman ceiling ~ sqrt(reliability): **{ceiling:.3f}**",
             f"- bootstrap resamples: {arguments.bootstrap}; stratification: k >= {arguments.k_min}\n"]

    modelResults = {}
    for checkpointPath in arguments.checkpoints:
        model, _ = loadCheckpoint(checkpointPath)
        # Annotate in-sample rows: checkpoints from trainMultiSize record trainedOn.sizes.
        rawCheckpoint = torch.load(checkpointPath, map_location="cpu", weights_only=False)
        trainedSizes = rawCheckpoint.get("trainedOn", {}).get("sizes")
        if trainedSizes is None:
            inSample = ""                                  # unknown (older checkpoints)
        else:
            inSample = (l, m) in {tuple(s) for s in trainedSizes}
        predictions = predictPerCode(model, data, errorRange = errorRange, 
                                     batchSize = 1024, 
                                     rewardEngineering = rewardEngineering)
        result = analyseModel(predictions, trueReward, trueK, arguments.k_min,
                              regretKs, resampleIndices)
        result["nll"] = binomialNllOfCurve(predictions["curve"], data.counts.astype(float),
                                           data.samples.astype(float))
        result["noiseFloor"] = noiseFloorNll(data.counts.astype(float),
                                             data.samples.astype(float))
        calRows, result["ece"] = calibrationTable(predictions["curve"],
                                                  data.counts, data.samples)
        modelResults[checkpointPath] = result

        name = checkpointPath
        sampleFlag = ("  **[IN-SAMPLE: this size is in the model's training data]**"
                      if inSample is True else "")
        lines += [f"\n## {name}{sampleFlag}\n",
                  f"- Spearman: {formatCi(result['spearman'])} "
                  f"(ceiling {ceiling:.3f})",
                  f"- Spearman on k>={arguments.k_min} stratum "
                  f"({result['stratumSize']} codes): {formatCi(result['spearmanStratum'])}"]
        for k in regretKs:
            lines.append(f"- regret@{k}: {formatCi(result['regret'][k], digits=5)} "
                         f"(reward units; 0 = surrogate's top-{k} contains the true best)")
        lines += [f"- k MAE: **{result['kMae']:.2f}**",
                  f"- reward MAE: **{result['maeBefore']:.5f}** -> "
                  f"**{result['maeAfter']:.5f}** after affine recalibration "
                  f"(slope {result['affineSlope']:.3f})",
                  f"- Binomial NLL: **{result['nll']:.4f}** "
                  f"(noise floor {result['noiseFloor']:.4f}); ECE: **{result['ece']:.5f}**",
                  "- calibration (predicted -> empirical, weight):"]
        for meanPredicted, empirical, weightFraction in calRows:
            lines.append(f"    - {meanPredicted:.4f} -> {empirical:.4f}  "
                         f"(w={weightFraction:.2f})")

        appendCsvRow(csvPath, {
            "date": today, "checkpoint": checkpointPath, "dataRoot": dataRoot,
            "l": l, "m": m, "numberOfCodes": numberOfCodes,
            "kMin": arguments.k_min, "inSample": inSample,
            "spearman": result["spearman"]["value"],
            "spearmanLow": result["spearman"]["low"],
            "spearmanHigh": result["spearman"]["high"],
            "spearmanStratum": result["spearmanStratum"]["value"],
            "spearmanStratumLow": result["spearmanStratum"]["low"],
            "spearmanStratumHigh": result["spearmanStratum"]["high"],
            "stratumSize": result["stratumSize"],
            **{f"regret{k}{suffix}": result["regret"][k][key]
               for k in (1, 5, 10) if k in result["regret"]
               for suffix, key in (("", "value"), ("Low", "low"), ("High", "high"))},
            "kMae": result["kMae"], "maeBefore": result["maeBefore"],
            "maeAfter": result["maeAfter"], "affineSlope": result["affineSlope"],
            "nll": result["nll"], "noiseFloor": result["noiseFloor"], "ece": result["ece"],
            "labelReliability": reliability, "labelCeiling": ceiling,
        }, CSV_FIELDS)

    if len(arguments.checkpoints) >= 2:
        lines.append("\n## Paired comparisons (Spearman difference on shared resamples)\n")
        for indexA in range(len(arguments.checkpoints)):
            for indexB in range(indexA + 1, len(arguments.checkpoints)):
                pathA, pathB = arguments.checkpoints[indexA], arguments.checkpoints[indexB]
                paired = pairedComparison(modelResults[pathA], modelResults[pathB])
                verdict = ("A better" if paired["low"] > 0 else
                           "B better" if paired["high"] < 0 else "not distinguishable")
                lines.append(f"- A=`{pathA}` vs B=`{pathB}`: "
                             f"diff {paired['value']:+.3f} "
                             f"[{paired['low']:+.3f}, {paired['high']:+.3f}], "
                             f"P(A>B)={paired['pABetter']:.2f} -> **{verdict}**")

    text = "\n".join(lines) + "\n"
    appendMarkdown(reportPath, text)
    print(text)
    print(f"appended to {reportPath}\ncsv row(s) appended to {csvPath}")


if __name__ == "__main__":
    main()
