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
from qecc.codeSurrogate import CodeCurvePredictor, binomialCurveLoss


def evaluateLoss(model, bits, counts, samples, k, l, m, batchSize=1024):
    model.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, bits.shape[0], batchSize):
            end = start + batchSize
            logits = model(bits[start:end], l, m, k[start:end])
            losses.append(binomialCurveLoss(logits, counts[start:end],
                                            samples[start:end]).item())
    model.train()
    return float(np.mean(losses))


def trainModel(data, epochs, batchSize, lr, seed,
               dModel=64, nHead=4, numLayers=2, dimFeedforward=128):
    torch.manual_seed(seed)
    train, val, test = splitData(data, fractions=(0.8, 0.1, 0.1), seed=seed)
    trainBits, trainCounts, trainSamples, trainK = toTensors(train)
    valBits, valCounts, valSamples, valK = toTensors(val)
    if trainBits.shape[0] == 0 or valBits.shape[0] == 0:
        raise ValueError(
            f"Dataset too small for a (train, val) split: got {trainBits.shape[0]} "
            f"train and {valBits.shape[0]} val rows. Provide more records or adjust fractions.")
    model = CodeCurvePredictor(dModel=dModel, nHead=nHead, numLayers=numLayers,
                               dimFeedforward=dimFeedforward)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"trainLoss": [], "valLoss": []}
    numberOfRecords = trainBits.shape[0]
    for epoch in range(epochs):
        permutation = torch.randperm(numberOfRecords)
        epochLosses = []
        for start in range(0, numberOfRecords, batchSize):
            batch = permutation[start:start + batchSize]
            optimizer.zero_grad()
            logits = model(trainBits[batch], data.l, data.m, trainK[batch])
            loss = binomialCurveLoss(logits, trainCounts[batch], trainSamples[batch])
            loss.backward()
            optimizer.step()
            epochLosses.append(loss.item())
        history["trainLoss"].append(float(np.mean(epochLosses)))
        history["valLoss"].append(evaluateLoss(model, valBits, valCounts,
                                               valSamples, valK, data.l, data.m))
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
    parser.add_argument("--checkpoint", default=None,
                        help="Defaults to $QECC_DATA/surrogate_<l>x<m>.pth")
    parser.add_argument("--report", default=os.path.join(
        os.environ.get("QECC_DATA", "C:/Users/Omer/rl-qecc-data"),
        "surrogate-transfer-report.md"))
    arguments = parser.parse_args()

    checkpointPath = arguments.checkpoint or os.path.join(
        os.environ.get("QECC_DATA", "C:/Users/Omer/rl-qecc-data"),
        f"surrogate_{arguments.l}x{arguments.m}.pth")

    data = loadCodeEvaluations(arguments.data_root, arguments.l, arguments.m)
    print(f"loaded {data.bits.shape[0]} unique codes at l={arguments.l}, m={arguments.m}")
    model, history, testData = trainModel(data, arguments.epochs, arguments.batch_size,
                                          arguments.lr, arguments.seed, arguments.d_model,
                                          arguments.n_head, arguments.num_layers,
                                          arguments.dim_feedforward)
    os.makedirs(os.path.dirname(checkpointPath) or ".", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "hyperParameters": {"dModel": arguments.d_model, "nHead": arguments.n_head,
                            "numLayers": arguments.num_layers,
                            "dimFeedforward": arguments.dim_feedforward,
                            "numberOfHarmonics": 3},
        "history": history,
        "trainedOn": {"l": arguments.l, "m": arguments.m, "seed": arguments.seed},
    }, checkpointPath)
    print(f"saved {checkpointPath}")

    # Evaluate on the held-out test split (10% of the real logged codes, never
    # seen in training or per-epoch monitoring) and append it to the report.
    from evaluateTransfer import evaluateOnData, appendReport
    testMetrics = evaluateOnData(model, testData)
    title = (f"{checkpointPath} on held-out test l={arguments.l}, m={arguments.m} "
             f"({testMetrics['numberOfCodes']} codes) — {datetime.date.today().isoformat()}")
    print(appendReport(arguments.report, title,
                       f"held-out 10% of {arguments.data_root}", testMetrics))


if __name__ == "__main__":
    main()
