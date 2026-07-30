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
from qecc.utils import calculateRewardFromSamples
from qecc.codeEvaluationDataset import (loadCodeEvaluations, toTensors)
from qecc.utils import NAMED_ERROR_RANGES #, GEOMETRIC5_ERROR_RANGE
from qecc.codeSurrogate import CodeCurvePredictor
import fnmatch
from qecc.utils import baselines
import copy
from trainSurrogate import resolveDevice

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


def evaluateOnData(model, data, errorRange, batchSize = 1024, rewardEngineering = True):
    device = next(model.parameters()).device # Hack -  Run wherever the model already lives. Inferring the device instead of taking it as
    # a parameter means no call site has to change, and it can never disagree with the model's own placement.
    bits, _counts, _samples, _k = toTensors(data)
    curves = []
    kPredictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, bits.shape[0], batchSize):
            end = start + batchSize
            chunk = bits[start:end].to(device, non_blocking=True)
            curveLogits, kLogPrediction = model(chunk, data.l, data.m)
            curves.append(torch.sigmoid(curveLogits).cpu())
            kPredictions.append(torch.expm1(kLogPrediction).cpu())
    curve = torch.cat(curves).numpy()
    kPredicted = torch.cat(kPredictions).numpy()

    trueReward = calculateRewardFromSamples(data.counts, 
                                            data.samples[:, None], #The None in data.samples[:, None] is required because counts is (N, 5) 
                                            errorRange=errorRange, 
                                            l = data.l, m = data.m, rewardEngineering=rewardEngineering)
    predictedReward = calculateRewardFromSamples(curve, numberOfSamples=1, # Important ! the curve predicted by the model should already represent error RATE !!!
                                                errorRange=errorRange,
                                                l = data.l, m = data.m, rewardEngineering=rewardEngineering)
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


def loadCheckpoint(path, device = "cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = CodeCurvePredictor(**checkpoint["hyperParameters"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device), checkpoint


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



def evaluateSweep(reportName, modelCheckpointsPath, dataPath, codeSizes, errorRange, rewardEngineering, device = "auto", batchSize = 8192):
    """
    Basically the same as what main does, just optimised towards loading the data once per code size, instead of per model
    Warning 1: The evaluation data is expected at dataPath/l_{l}_m_{m}
    Warning 2:
    """
    modelCheckpoints =[]
    for dirpath, dirnames, filenames in os.walk(modelCheckpointsPath):
            modelCheckpoints.extend(os.path.join(dirpath, m) for m in fnmatch.filter(filenames, "*.pth"))
    modelCheckpoints.sort()
    if not os.path.isdir(os.environ.get("QECC_DATA")):
        raise ValueError("Environment variable QECC_DATA must be set.")
    reportPath = os.path.join(os.environ.get("QECC_DATA"), reportName)
    csvReportPath = os.path.join(os.environ.get("QECC_DATA"), f"{reportName}.csv")
    reportTemplate = {"Model checkpoint":None,
                    "Number of harmonics":None,
                    "seed":None,
                    "epochs":None,
                    "recipie":None,
                    "trained on sizes": None,
                    "6,6": False,
                    "9,6": False,
                    "12,6": False,
                    "15,3": False,
                    "error range name":errorRange,
                    "numberOfCodes":None,
                    "code l":None,
                    "code m":None,
                    "Date":None,}
                    # "nll":None, 
                    # "noiseFloor":None, 
                    # "rewardMae":None, 
                    # "kMae":None,
                    # "spearman":None,
                    # "kendall":None,
                    # "topK50":None,}
    
    firstRun = (not os.path.exists(csvReportPath)) or os.path.getsize(csvReportPath) == 0
    device = resolveDevice(device)
    print(f"evaluateSweep on {device}: {len(modelCheckpoints)} checkpoints x {len(list(codeSizes))} sizes")
    
    errorRange = NAMED_ERROR_RANGES[errorRange]
    
    
    for (l,m) in codeSizes:
        # Note the order here !!! 2. Load the data, i.e., the code records that contain codes with parameters l,m and CONTAIN the relevant error range (could be they have MORE - can't be less -  error points, if yes filter out just the relevant error points)
        sizeSubfolder = os.path.join(dataPath, f"l_{l}_m_{m}") # Optimization - don't scan ALL the data path, just the relevant subfolder.
        if not os.path.isdir(sizeSubfolder):
            raise ValueError(f"Expected data to be found in {dataPath}/l_{l}_m_{m}, but not such folder was found. ")
        data = loadCodeEvaluations(sizeSubfolder, l, m, errorRange=errorRange)
        
        for checkPoint in modelCheckpoints:
            report = copy.deepcopy(reportTemplate)
            report["code l"], report["code m"] = l, m
            report["Date"] = datetime.date.today().isoformat()
            # 1. Load model from checkpoint, instantiate a code curve and k predictor, set their weights to the checkpointed weghts.
            report["Model checkpoint"] = checkPoint
            model,ck = loadCheckpoint(checkPoint, device = device)
            report["Number of harmonics"] = ck["hyperParameters"]["numberOfHarmonics"]
            trainedOn = ck["trainedOn"]
            report["seed"] = trainedOn["seed"]
            report["epochs"] = str(len(ck["history"]["trainLoss"]))      # not stored directly
            report["recipie"] = ("mixed" if trainedOn["initCheckpoint"] is None
             else "groundPure" if not trainedOn["curveSizes"]
             else "groundAnchored")
            report["trained on sizes"] = trainedOn["sizes"]
            trainingSizes = [tuple(s) for s in trainedOn["sizes"]] 
            for t in trainingSizes:
                report[f"{t[0]},{t[1]}"] = True 
            # 3. Evaluate the loaded model on the loaded data. 
            #   Get the estimation for k and the error curve from the model, calculate the predicted reward. Compare with the true k and reward from the data.
            metrics = evaluateOnData(model, data, batchSize=batchSize, errorRange=errorRange, rewardEngineering=rewardEngineering)
            del model # Free up VRAM
            #title = (f"{checkPoint} on l={l}, m={m} "
            #        f"({metrics['numberOfCodes']} codes) — {datetime.date.today().isoformat()}")
            #4. this is the part I don't like, it appends the results to a markdown file at the moment, I need to move this to pandas.
            #print(appendReport(reportPath, title, dataPath, metrics))
            for key, value in metrics.items():
                report[key] = value
            newRow = "\t".join(str(report[k]) for k in report)
            if firstRun:
                with open(csvReportPath, 'a') as fid:
                    fid.write("\t".join(key for key in report.keys()))
                    fid.write("\n")
                    fid.flush() 
                    firstRun = False
            with open(csvReportPath, 'a') as fid:
                fid.write(newRow+"\n")
                fid.flush()    
    return
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--l", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--error-range", type=str, required=True, choices = sorted(NAMED_ERROR_RANGES))
    parser.add_argument("--report", default=None,
                        help="Defaults to surrogate-transfer-report.md next to the checkpoint")
    parser.add_argument("--reward-engineering", type = str, default="true", choices=["true", "false", "True", "False"])
    arguments = parser.parse_args()
    # Default: append to the report in the checkpoint's own run folder, so a
    # training run and its cross-size transfer evaluations share one directory.
    reportPath = arguments.report or os.path.join(
        os.path.dirname(arguments.checkpoint) or ".", "surrogate-transfer-report.md")
    errorRange = NAMED_ERROR_RANGES[arguments.error_range]
    rewardEngineering = arguments.reward_engineering.lower() == "true"
    # 1. Load model from checkpoint, instantiate a code curve and k predictor, set their weights to the checkpointed weghts.
    model, ck = loadCheckpoint(arguments.checkpoint)
    # 2. Load the data, i.e., the code records that contain codes with parameters l,m and CONTAIN the relevant error range (could be they have MORE - can't be less -  error points, if yes filter out just the relevant error points)
    data = loadCodeEvaluations(arguments.data_root, arguments.l, arguments.m, errorRange=errorRange)
    # 3. Evaluate the loaded model on the loaded data. 
    #   Get the estimation for k and the error curve from the model, calculate the predicted reward. Compare with the true k and reward from the data.
    metrics = evaluateOnData(model, data, batchSize=1024, errorRange=errorRange, rewardEngineering=rewardEngineering)
    title = (f"{arguments.checkpoint} on l={arguments.l}, m={arguments.m} "
             f"({metrics['numberOfCodes']} codes) — {datetime.date.today().isoformat()}")
    #4. this is the part I don't like, it appends the results to a markdown file at the moment, I need to move this to pandas.
    print(appendReport(reportPath, title, arguments.data_root, metrics))


if __name__ == "__main__":
    #main()
    dataPath = os.environ.get("QECC_DATA")
    if dataPath is None:
            raise ValueError("Environment variable QECC_DATA must be set and point at the data.")
    dataPath = os.path.join(dataPath, "supervisedLearning")
    modelsPath = os.path.join(dataPath, "sweep2")
    dataPath = os.path.join(dataPath, "codeEvaluationTrainingData")
    evaluateSweep("geometric5Sweep", modelsPath, dataPath, codeSizes = sorted(baselines), errorRange="geometric5", rewardEngineering=True, device = "auto", batchSize = 8192)
    

