"""Loader for codeEvaluations_*.jsonl decoder-evaluation logs.

Each record is one (code -> BER curve) supervision pair written by
bb_gym_v_0_1.logCodeEvaluation. This module discovers, filters, dedups, and
tensorizes them for surrogate training. See
docs/superpowers/specs/2026-07-11-code-evaluation-surrogate-design.md.
"""
import fnmatch
import json
import os
from dataclasses import dataclass

import numpy as np
import torch

CANONICAL_ERROR_RANGE = np.linspace(0.0001, 0.1, 5)
GRID_TOLERANCE = 1e-9


@dataclass
class CodeEvaluationData:
    bits: np.ndarray      # (N, 2l+2m) int8, order [aX | aY | bX | bY]
    counts: np.ndarray    # (N, 5) int64, combined logical + decoder-failure counts
    samples: np.ndarray   # (N,) int64, total Bernoulli trials (post-dedup)
    k: np.ndarray         # (N,) int64, numberOfLogicalQubits
    l: int
    m: int


def _iterRecords(rootDirectory):
    for dirpath, _dirnames, filenames in os.walk(rootDirectory):
        for name in fnmatch.filter(filenames, "codeEvaluations_*.jsonl"):
            with open(os.path.join(dirpath, name)) as fid:
                for line in fid:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue  # tolerate a truncated last line from a killed run


def _foldCoefficients(values, period):
    """Fold an over-length coefficient vector to `period` via x^period = 1 (XOR over residues)."""
    folded = np.zeros(period, dtype=np.int8)
    for i, bit in enumerate(values):
        if bit:
            folded[i % period] ^= 1
    return folded


def loadCodeEvaluations(rootDirectory, l, m, errorRange=CANONICAL_ERROR_RANGE):
    """Load, filter to (l, m) and the canonical grid, dedup by summing counts."""
    aggregated = {}  # bits tuple -> [counts (5,), samples, k]
    dropped = 0
    for record in _iterRecords(rootDirectory):
        if record["l"] != l or record["m"] != m:
            continue
        recordGrid = np.asarray(record["errorRange"], dtype=float)
        if recordGrid.shape != np.shape(errorRange) or \
                not np.allclose(recordGrid, errorRange, atol=GRID_TOLERANCE):
            dropped += 1
            continue
        foldedAX = _foldCoefficients(record["aX"], l)
        foldedBX = _foldCoefficients(record["bX"], l)
        foldedAY = _foldCoefficients(record["aY"], m)
        foldedBY = _foldCoefficients(record["bY"], m)
        bits = tuple(foldedAX) + tuple(foldedAY) + tuple(foldedBX) + tuple(foldedBY)
        combined = np.asarray(record["logicalErrorCounts"], dtype=np.int64) + \
                   np.asarray(record["decoderFailureCounts"], dtype=np.int64)
        entry = aggregated.get(bits)
        if entry is None:
            aggregated[bits] = [combined,
                                int(record["numberOfSamples"]),
                                int(record["numberOfLogicalQubits"])]
        else:
            entry[0] = entry[0] + combined
            entry[1] = entry[1] + int(record["numberOfSamples"])
    if dropped:
        print(f"loadCodeEvaluations: dropped {dropped} records with a non-canonical errorRange")
    if not aggregated:
        raise ValueError(f"No records found for l={l}, m={m} under {rootDirectory}")
    bitsArray = np.array(list(aggregated.keys()), dtype=np.int8)
    countsArray = np.stack([entry[0] for entry in aggregated.values()])
    samplesArray = np.array([entry[1] for entry in aggregated.values()], dtype=np.int64)
    kArray = np.array([entry[2] for entry in aggregated.values()], dtype=np.int64)
    return CodeEvaluationData(bitsArray, countsArray, samplesArray, kArray, l, m)


def rewardFromCounts(counts, samples, errorRange):
    """Reconstruct the environment reward: trapezoid(1 - combinedBER, errorRange)."""
    ber = counts / samples[:, None]
    return np.trapezoid(1.0 - ber, np.asarray(errorRange, dtype=float), axis=-1)


def _subset(data, indices):
    return CodeEvaluationData(data.bits[indices], data.counts[indices],
                              data.samples[indices], data.k[indices], data.l, data.m)


def splitData(data, fractions=(0.8, 0.1, 0.1), seed=0):
    assert abs(sum(fractions) - 1.0) < 1e-9, "fractions must sum to 1"
    n = data.bits.shape[0]
    permutation = np.random.default_rng(seed).permutation(n)
    trainEnd = int(round(fractions[0] * n))
    valEnd = trainEnd + int(round(fractions[1] * n))
    return (_subset(data, permutation[:trainEnd]),
            _subset(data, permutation[trainEnd:valEnd]),
            _subset(data, permutation[valEnd:]))


def toTensors(data):
    return (torch.as_tensor(data.bits, dtype=torch.float32),
            torch.as_tensor(data.counts, dtype=torch.float32),
            torch.as_tensor(data.samples, dtype=torch.float32),
            torch.as_tensor(data.k, dtype=torch.float32))
