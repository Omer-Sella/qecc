# surrogate/trainMultiSize.py
"""Multi-size surrogate training with size-homogeneous minibatches.

One loop covers both experiments from the 2026-07-22 transfer results:

FULL MIXED training (curve + k loss on every size):
    python surrogate/trainMultiSize.py --sizes 6,6 9,6 \
        --data-root $QECC_DATA/supervisedLearning/codeEvaluationTrainingData

k-ONLY GROUNDING of an existing checkpoint (k loss everywhere, curve loss nowhere;
k labels are exact at any size, so this needs no decoder quality at the new sizes):
    python surrogate/trainMultiSize.py --sizes 6,6 9,6 12,6 15,3 --curve-sizes none \
        --init-checkpoint $QECC_DATA/supervisedLearning/2026-07-12_21-15-58/surrogate_6x6.pth \
        --data-root $QECC_DATA/supervisedLearning/codeEvaluationTrainingData

k-grounding WITH a 6,6 curve anchor (recommended guard against curve-head drift):
    ... --curve-sizes 6,6 ...

Different sizes have different bit widths and period structure, so every minibatch is
drawn from ONE size; sizes are interleaved and shuffled within each epoch. Sequential
training (all of size A, then all of size B) is what this loop deliberately avoids:
it invites catastrophic forgetting of A. If you want to MEASURE that forgetting, chain
two runs via --init-checkpoint and compare the per-size held-out blocks.

After training: per-size held-out (10%) evaluation is appended to the run report,
including the 6,6 block — that is the regression check that grounding did not damage
the within-size surrogate.
"""
import argparse
import datetime
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trainSurrogate import resolveDevice  # noqa: E402
from evaluateTransfer import loadCheckpoint, evaluateOnData, appendReport  # noqa: E402

from qecc.codeEvaluationDataset import (loadCodeEvaluations, splitData, toTensors,  # noqa: E402
                                        _subset)
from qecc.codeSurrogate import (CodeCurvePredictor, binomialCurveLoss,  # noqa: E402
                                kPredictionLoss)


def parseSizes(tokens):
    if tokens is None:
        return []
    if len(tokens) == 1 and tokens[0].lower() == "none":
        return []
    return [tuple(int(part) for part in token.split(",")) for token in tokens]


def loadSizeData(dataRoot, l, m, maxCodes, seed):
    sizeSubfolder = os.path.join(dataRoot, f"l_{l}_m_{m}")
    root = sizeSubfolder if os.path.isdir(sizeSubfolder) else dataRoot
    data = loadCodeEvaluations(root, l, m)
    if maxCodes and data.bits.shape[0] > maxCodes:
        keep = np.random.default_rng(seed).choice(data.bits.shape[0], maxCodes, replace=False)
        data = _subset(data, keep)
    return data


class SizeSlot:
    """Everything one (l, m) size contributes to the loop, tensors on device."""

    def __init__(self, data, device, seed, useCurveLoss):
        self.l, self.m = data.l, data.m
        self.name = f"{data.l},{data.m}"
        self.useCurveLoss = useCurveLoss
        self.train, self.val, self.test = splitData(data, fractions=(0.8, 0.1, 0.1), seed=seed)
        if self.train.bits.shape[0] == 0 or self.val.bits.shape[0] == 0:
            raise ValueError(f"size {self.name}: dataset too small for a (train, val) split")
        (self.trainBits, self.trainCounts,
         self.trainSamples, self.trainK) = (t.to(device) for t in toTensors(self.train))
        (self.valBits, self.valCounts,
         self.valSamples, self.valK) = (t.to(device) for t in toTensors(self.val))

    def numberOfTrainRows(self):
        return self.trainBits.shape[0]


def maskedLoss(model, slot, rows, kLossWeight):
    """The per-size objective: k loss always; curve loss only where configured."""
    curveLogits, kLogPrediction = model(slot.trainBits[rows], slot.l, slot.m)
    loss = kLossWeight * kPredictionLoss(kLogPrediction, slot.trainK[rows])
    if slot.useCurveLoss:
        loss = loss + binomialCurveLoss(curveLogits, slot.trainCounts[rows],
                                        slot.trainSamples[rows])
    return loss


def validationLoss(model, slot, kLossWeight, batchSize=1024):
    model.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, slot.valBits.shape[0], batchSize):
            end = start + batchSize
            curveLogits, kLogPrediction = model(slot.valBits[start:end], slot.l, slot.m)
            loss = kLossWeight * kPredictionLoss(kLogPrediction, slot.valK[start:end])
            if slot.useCurveLoss:
                loss = loss + binomialCurveLoss(curveLogits, slot.valCounts[start:end],
                                                slot.valSamples[start:end])
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def buildEpochSchedule(slots, batchSize, weighting, rng, device):
    """List of (slotIndex, rowIndices) minibatches; each batch from ONE size; order shuffled.

    balanced: every size contributes the same number of batches (set by the smallest
    size), drawing a fresh random subset of the bigger sizes each epoch.
    proportional: every size contributes ceil(n/batchSize) batches (big sizes dominate).
    """
    perSizeBatchCounts = []
    if weighting == "balanced":
        smallest = min(slot.numberOfTrainRows() for slot in slots)
        batchesEach = max(1, int(np.ceil(smallest / batchSize)))
        perSizeBatchCounts = [batchesEach] * len(slots)
    else:
        perSizeBatchCounts = [max(1, int(np.ceil(slot.numberOfTrainRows() / batchSize)))
                              for slot in slots]
    schedule = []
    for slotIndex, (slot, numberOfBatches) in enumerate(zip(slots, perSizeBatchCounts)):
        permutation = torch.randperm(slot.numberOfTrainRows(), device=device)
        for batchIndex in range(numberOfBatches):
            rows = permutation[batchIndex * batchSize:(batchIndex + 1) * batchSize]
            if len(rows) > 0:
                schedule.append((slotIndex, rows))
    rng.shuffle(schedule)
    return schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True,
                        help="Base folder holding l_{l}_m_{m}/ subfolders "
                             "(codeEvaluationTrainingData).")
    parser.add_argument("--sizes", nargs="+", required=True, metavar="L,M",
                        help="Sizes to train on, e.g. --sizes 6,6 9,6")
    parser.add_argument("--curve-sizes", nargs="+", default=None, metavar="L,M",
                        help="Sizes that ALSO get the curve loss. Default: all of --sizes "
                             "(full mixed training). 'none' = k-only grounding.")
    parser.add_argument("--init-checkpoint", default=None,
                        help="Warm-start from this surrogate checkpoint (else fresh model).")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=None,
                        help="Default: 1e-4 when fine-tuning from a checkpoint, else 1e-3.")
    parser.add_argument("--k-loss-weight", type=float, default=1.0)
    parser.add_argument("--size-weighting", choices=["balanced", "proportional"],
                        default="balanced",
                        help="balanced: equal batches per size per epoch (recommended; "
                             "1.3M 6,6 codes would otherwise drown 17k 15,3 codes).")
    parser.add_argument("--max-codes-per-size", type=int, default=0,
                        help="Random cap per size (0 = all); useful for quick tests.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", default=None,
                        help="Output path; default $QECC_DATA/supervisedLearning/<timestamp>/"
                             "surrogate_multi.pth")
    parser.add_argument("--report", default=None)
    arguments = parser.parse_args()

    sizes = parseSizes(arguments.sizes)
    curveSizes = set(parseSizes(arguments.curve_sizes) if arguments.curve_sizes is not None
                     else sizes)
    unknownCurveSizes = curveSizes - set(sizes)
    if unknownCurveSizes:
        raise SystemExit(f"--curve-sizes {sorted(unknownCurveSizes)} not in --sizes {sizes}")
    learningRate = arguments.lr if arguments.lr is not None else (
        1e-4 if arguments.init_checkpoint else 1e-3)

    device = resolveDevice(arguments.device)
    torch.manual_seed(arguments.seed)
    rng = np.random.default_rng(arguments.seed)
    print(f"device: {device}; sizes: {sizes}; curve loss on: {sorted(curveSizes) or 'NONE'}; "
          f"lr: {learningRate}")

    slots = []
    for l, m in sizes:
        data = loadSizeData(arguments.data_root, l, m, arguments.max_codes_per_size,
                            arguments.seed)
        slot = SizeSlot(data, device, arguments.seed, useCurveLoss=(l, m) in curveSizes)
        print(f"  size {slot.name}: {data.bits.shape[0]} codes "
              f"({slot.numberOfTrainRows()} train) curveLoss={slot.useCurveLoss}")
        slots.append(slot)

    if arguments.init_checkpoint:
        model = loadCheckpoint(arguments.init_checkpoint).to(device)
        checkpoint = torch.load(arguments.init_checkpoint, map_location="cpu",
                                weights_only=False)
        hyperParameters = checkpoint["hyperParameters"]
        print(f"warm-started from {arguments.init_checkpoint}")
    else:
        hyperParameters = {"dModel": arguments.d_model, "nHead": arguments.n_head,
                           "numLayers": arguments.num_layers,
                           "dimFeedforward": arguments.dim_feedforward,
                           "numberOfHarmonics": 3}
        model = CodeCurvePredictor(**hyperParameters).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

    history = {"trainLoss": [], "valLoss": {slot.name: [] for slot in slots}}
    for epoch in range(arguments.epochs):
        schedule = buildEpochSchedule(slots, arguments.batch_size,
                                      arguments.size_weighting, rng, device)
        epochLosses = []
        for slotIndex, rows in schedule:
            optimizer.zero_grad()
            loss = maskedLoss(model, slots[slotIndex], rows, arguments.k_loss_weight)
            loss.backward()
            optimizer.step()
            epochLosses.append(loss.item())
        history["trainLoss"].append(float(np.mean(epochLosses)))
        valSummary = []
        for slot in slots:
            valLoss = validationLoss(model, slot, arguments.k_loss_weight)
            history["valLoss"][slot.name].append(valLoss)
            valSummary.append(f"{slot.name}: {valLoss:.4f}")
        print(f"epoch {epoch}: train {history['trainLoss'][-1]:.4f} | "
              f"val {' | '.join(valSummary)}", flush=True)

    runDirectory = os.path.join(
        os.environ.get("QECC_DATA", "C:/Users/Omer/rl-qecc-data"), "supervisedLearning",
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    checkpointPath = arguments.checkpoint or os.path.join(runDirectory, "surrogate_multi.pth")
    reportPath = arguments.report or os.path.join(
        os.path.dirname(checkpointPath), "surrogate-transfer-report.md")

    model = model.cpu()
    os.makedirs(os.path.dirname(checkpointPath) or ".", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "hyperParameters": hyperParameters,
        "history": history,
        "trainedOn": {"sizes": sizes, "curveSizes": sorted(curveSizes),
                      "initCheckpoint": arguments.init_checkpoint,
                      "seed": arguments.seed, "kLossWeight": arguments.k_loss_weight,
                      "sizeWeighting": arguments.size_weighting, "lr": learningRate},
    }, checkpointPath)
    print(f"saved {checkpointPath}")

    # Self-describing report: record the run configuration so a block can never be
    # attributed to the wrong experiment even if the checkpoint filename is misleading.
    os.makedirs(os.path.dirname(reportPath) or ".", exist_ok=True)
    with open(reportPath, "a", encoding="utf-8") as fid:
        fid.write(f"\n## {checkpointPath} — run configuration "
                  f"({datetime.date.today().isoformat()})\n\n"
                  f"- sizes: {sizes}; curve loss on: {sorted(curveSizes) or 'NONE'}\n"
                  f"- initCheckpoint: {arguments.init_checkpoint or 'fresh model'}\n"
                  f"- epochs: {arguments.epochs}, lr: {learningRate}, "
                  f"seed: {arguments.seed}, weighting: {arguments.size_weighting}, "
                  f"kLossWeight: {arguments.k_loss_weight}\n")

    # Per-size held-out evaluation, INCLUDING the within-6,6 regression check.
    for slot in slots:
        metrics = evaluateOnData(model, slot.test)
        title = (f"{checkpointPath} on held-out test l={slot.l}, m={slot.m} "
                 f"({metrics['numberOfCodes']} codes) — {datetime.date.today().isoformat()}")
        print(appendReport(reportPath, title,
                           f"held-out 10% of {arguments.data_root} (size {slot.name})",
                           metrics))


if __name__ == "__main__":
    main()
