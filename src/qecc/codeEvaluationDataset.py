"""
Loader for codeEvaluations_*.jsonl decoder-evaluation logs.

This module discovers, filters, deduplicates, and tensorizes the records them for surrogate training.

"""
import fnmatch
import json
import os
from dataclasses import dataclass

import numpy as np
import torch

#from qecc.utils import CANONICAL_ERROR_RANGE
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


def _gridProjection(recordGrid, errorRange, tolerance=GRID_TOLERANCE):
    """Column indices of errorRange inside recordGrid, or None if recordGrid lacks a point."""
    columns = []
    for point in errorRange:
        hits = np.flatnonzero(np.isclose(recordGrid, point, rtol=0, atol=tolerance))
        if hits.size != 1:
            return None
        columns.append(int(hits[0]))
    if not len(columns) == len(errorRange):
        raise ValueError("You plugged an error range that is only partially found in the records. This guard is to make sure you don't accidentally ask for 5 error points and get 3 matching. If this was intended you would have to manually remove this safeguard.")
    return columns

def loadCodeEvaluations(rootDirectory, l, m, errorRange):
    """Load, filter to (l, m) and the specified errorRange, dedup by summing counts."""
    aggregated = {}  # bits tuple -> [counts (5,), samples, k]
    dropped = 0
    projections = {}
    for record in _iterRecords(rootDirectory):
        if record["l"] != l or record["m"] != m:
            continue
        recordGrid = np.asarray(record["errorRange"], dtype=float)

        gridKey = recordGrid.tobytes() # We're checking if the error range (grid) in the record has already been parsed in some past record. If yes, we use the projection already computed. If no, we calculate a projection
        if gridKey not in projections:
            projections[gridKey] = _gridProjection(recordGrid, errorRange)
        columns = projections[gridKey]
        if columns is None:
            dropped += 1
            continue
        combined = (np.asarray(record["logicalErrorCounts"], dtype=np.int64)[columns] +
                    np.asarray(record["decoderFailureCounts"], dtype=np.int64)[columns])

        
        bits = tuple(record["aX"]) + tuple(record["aY"]) + tuple(record["bX"]) + tuple(record["bY"])
        entry = aggregated.get(bits)
        if entry is None:
            aggregated[bits] = [combined,
                                int(record["numberOfSamples"]),
                                int(record["numberOfLogicalQubits"])]
        else:
            entry[0] = entry[0] + combined
            entry[1] = entry[1] + int(record["numberOfSamples"])
    if dropped:
        print(f"loadCodeEvaluations: kept {len(aggregated)} codes on the requested grid; "
              f"dropped {dropped} records whose grid does not contain all its points")
    if not aggregated:
        raise ValueError(f"No records found for l={l}, m={m} under {rootDirectory}")
    bitsArray = np.array(list(aggregated.keys()), dtype=np.int8)
    countsArray = np.stack([entry[0] for entry in aggregated.values()])
    samplesArray = np.array([entry[1] for entry in aggregated.values()], dtype=np.int64)
    kArray = np.array([entry[2] for entry in aggregated.values()], dtype=np.int64)
    return CodeEvaluationData(bitsArray, countsArray, samplesArray, kArray, l, m)


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
