# surrogate/trainSurrogate.py
"""Train the decoder surrogate on codeEvaluations jsonl data.

Usage:
    python surrogate/trainSurrogate.py --data-root C:/Users/Omer/rl-qecc-data \
        --l 6 --m 6 --epochs 20 --checkpoint surrogate/checkpoints/surrogate_6x6.pth
"""
import argparse
import datetime
import os

import numpy as np
import torch

from qecc.codeEvaluationDataset import loadCodeEvaluations, splitData, toTensors
from qecc.codeSurrogate import CodeCurvePredictor, binomialCurveLoss, kPredictionLoss
from qecc.utils import NAMED_ERROR_RANGES


def resolveDevice(requested):
    """'auto' -> cuda when available, else cpu. Explicit 'cuda' fails loudly if absent."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA was requested but torch.cuda.is_available() is False - "
            "check the node allocation / torch build, or use --device cpu.")
    return device


def combinedLoss(model, bits, counts, samples, k, l, m, kLossWeight):
    curveLogits, kLogPrediction = model(bits, l, m)
    return binomialCurveLoss(curveLogits, counts, samples) \
        + kLossWeight * kPredictionLoss(kLogPrediction, k)


def evaluateLoss(model, bits, counts, samples, k, l, m, kLossWeight, batchSize=1024):
    model.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, bits.shape[0], batchSize):
            end = start + batchSize
            losses.append(combinedLoss(model, bits[start:end], counts[start:end],
                                       samples[start:end], k[start:end],
                                       l, m, kLossWeight).item())
    model.train()
    return float(np.mean(losses))


def trainModel(data, epochs, batchSize, lr, seed,
               dModel=64, nHead=4, numLayers=2, dimFeedforward=128,
               kLossWeight=1.0, device="cpu"):
    device = torch.device(device)
    torch.manual_seed(seed)  # seeds CUDA generators too
    train, val, test = splitData(data, fractions=(0.8, 0.1, 0.1), seed=seed)
    trainBits, trainCounts, trainSamples, trainK = (
        t.to(device) for t in toTensors(train))
    valBits, valCounts, valSamples, valK = (
        t.to(device) for t in toTensors(val))
    if trainBits.shape[0] == 0 or valBits.shape[0] == 0:
        raise ValueError(
            f"Dataset too small for a (train, val) split: got {trainBits.shape[0]} "
            f"train and {valBits.shape[0]} val rows. Provide more records or adjust fractions.")
    model = CodeCurvePredictor(dModel=dModel, nHead=nHead, numLayers=numLayers,
                               dimFeedforward=dimFeedforward).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"trainLoss": [], "valLoss": []}
    numberOfRecords = trainBits.shape[0]
    for epoch in range(epochs):
        permutation = torch.randperm(numberOfRecords, device=device)
        epochLosses = []
        for start in range(0, numberOfRecords, batchSize):
            batch = permutation[start:start + batchSize]
            optimizer.zero_grad()
            loss = combinedLoss(model, trainBits[batch], trainCounts[batch],
                                trainSamples[batch], trainK[batch],
                                data.l, data.m, kLossWeight)
            loss.backward()
            optimizer.step()
            epochLosses.append(loss.item())
        history["trainLoss"].append(float(np.mean(epochLosses)))
        history["valLoss"].append(evaluateLoss(model, valBits, valCounts,
                                               valSamples, valK, data.l, data.m,
                                               kLossWeight))
        print(f"epoch {epoch}: train {history['trainLoss'][-1]:.4f} "
              f"val {history['valLoss'][-1]:.4f}", flush=True)
    return model, history, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.environ.get("QECC_DATA",
                                                              "C:/Users/Omer/rl-qecc-data"))
    parser.add_argument("--l", type=int, default=6)
    parser.add_argument("--m", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--k-loss-weight", type=float, default=1.0,
                        help="Weight of the auxiliary k-prediction loss")
    parser.add_argument("--device", default="auto",
                        help="'auto' (cuda if available, else cpu), 'cpu', 'cuda', or 'cuda:N'")
    parser.add_argument("--checkpoint", default=None,
                        help="Defaults to $QECC_DATA/supervisedLearning/<timestamp>/surrogate_<l>x<m>.pth")
    parser.add_argument("--report", default=None,
                        help="Defaults to surrogate-transfer-report.md in the same run folder")
    arguments = parser.parse_args()

    # Each training run gets its own timestamped folder under QECC_DATA,
    # matching the env logger's directory convention (e.g. 2026-07-12_09-30-00).
    runDirectory = os.path.join(
        os.environ.get("QECC_DATA", "C:/Users/Omer/rl-qecc-data"),
        "supervisedLearning",
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    checkpointPath = arguments.checkpoint or os.path.join(
        runDirectory, f"surrogate_{arguments.l}x{arguments.m}.pth")
    reportPath = arguments.report or os.path.join(runDirectory,
                                                  "surrogate-transfer-report.md")

    device = resolveDevice(arguments.device)
    print(f"training on device: {device}")

    data = loadCodeEvaluations(arguments.data_root, arguments.l, arguments.m)
    print(f"loaded {data.bits.shape[0]} unique codes at l={arguments.l}, m={arguments.m}")
    model, history, testData = trainModel(data, arguments.epochs, arguments.batch_size,
                                          arguments.lr, arguments.seed, arguments.d_model,
                                          arguments.n_head, arguments.num_layers,
                                          arguments.dim_feedforward,
                                          kLossWeight=arguments.k_loss_weight,
                                          device=device)
    # Back to CPU so the checkpoint is device-agnostic and the held-out
    # evaluation below (CPU tensors from toTensors) gets a matching model.
    model = model.cpu()
    os.makedirs(os.path.dirname(checkpointPath) or ".", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "hyperParameters": {"dModel": arguments.d_model, "nHead": arguments.n_head,
                            "numLayers": arguments.num_layers,
                            "dimFeedforward": arguments.dim_feedforward,
                            "numberOfHarmonics": 3},
        "history": history,
        "trainedOn": {"l": arguments.l, "m": arguments.m, "seed": arguments.seed,
                      "kLossWeight": arguments.k_loss_weight},
    }, checkpointPath)
    print(f"saved {checkpointPath}")

    # Evaluate on the held-out test split (10% of the real logged codes, never
    # seen in training or per-epoch monitoring) and append it to the report.
    from evaluateTransfer import evaluateOnData, appendReport
    testMetrics = evaluateOnData(model, testData, errorRange=NAMED_ERROR_RANGES[arguments.errorRange])
    title = (f"{checkpointPath} on held-out test l={arguments.l}, m={arguments.m} "
             f"({testMetrics['numberOfCodes']} codes) — {datetime.date.today().isoformat()}")
    print(appendReport(reportPath, title,
                       f"held-out 10% of {arguments.data_root}", testMetrics))


if __name__ == "__main__":
    main()
