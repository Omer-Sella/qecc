# tests/test_codeEvaluationDataset.py
import json
import numpy as np
import pytest
import torch
from qecc.utils import CANONICAL_ERROR_RANGE
from qecc.codeEvaluationDataset import (CodeEvaluationData,
    loadCodeEvaluations, splitData, toTensors,
)
from qecc.utils import calculateRewardFromSamples


def makeRecord(aX, aY, bX, bY, logical, failures,
               errorRange=None, numberOfSamples=50, k=8, seed=1):
    if errorRange is None:
        errorRange = [float(p) for p in CANONICAL_ERROR_RANGE]
    return {
        "timestamp": "2026-07-11T00:00:00+00:00", "runId": None,
        "l": len(aX), "m": len(aY),
        "aX": aX, "aY": aY, "bX": bX, "bY": bY,
        "numberOfLogicalQubits": k, "errorRange": errorRange,
        "logicalErrorCounts": logical, "decoderFailureCounts": failures,
        "numberOfSamples": numberOfSamples, "seed": seed,
        "decoder": "dual binary bposd 0", "noiseModel": "depolarizing",
    }


@pytest.fixture
def dataDirectory(tmp_path):
    codeA = dict(aX=[1, 0, 1, 0, 0, 0], aY=[0, 0, 0, 0, 0, 0],
                 bX=[0, 0, 0, 0, 0, 0], bY=[0, 1, 0, 0, 0, 0])
    codeB = dict(aX=[0, 0, 0, 1, 0, 0], aY=[0, 1, 0, 0, 0, 0],
                 bX=[0, 0, 1, 0, 0, 0], bY=[0, 0, 0, 0, 0, 1])
    records = [
        makeRecord(**codeA, logical=[0, 10, 20, 30, 40], failures=[0, 0, 0, 0, 5], seed=1),
        makeRecord(**codeA, logical=[0, 12, 18, 30, 40], failures=[0, 0, 2, 0, 5], seed=2),  # duplicate code
        makeRecord(**codeB, logical=[0, 5, 10, 20, 30], failures=[0, 0, 0, 0, 0], seed=3),
        # wrong grid -> must be dropped:
        makeRecord(**codeB, logical=[0, 1, 2, 3, 4], failures=[0, 0, 0, 0, 0],
                   errorRange=[0.001, 0.01, 0.02, 0.05, 0.1], seed=4),
    ]
    subdir = tmp_path / "2026-07-11_00-00-00"
    subdir.mkdir()
    with open(subdir / "codeEvaluations_1234.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return tmp_path


def test_loadFiltersGridAndDedups(dataDirectory):
    data = loadCodeEvaluations(str(dataDirectory), l=6, m=6, errorRange=CANONICAL_ERROR_RANGE)
    assert isinstance(data, CodeEvaluationData)
    assert data.bits.shape == (2, 24)          # 2 unique codes; bad-grid record dropped
    assert data.counts.shape == (2, 5)
    duplicatedRow = data.samples.argmax()
    assert data.samples[duplicatedRow] == 100  # 50 + 50 summed
    # combined counts of the duplicated code: logical+failures summed across both records
    expected = np.array([0, 22, 40, 60, 90])   # (0+0+0+0, 10+12, 20+18+2, 30+30, 40+5+40+5)
    np.testing.assert_array_equal(data.counts[duplicatedRow], expected)


def test_bitOrderIsAxAyBxBy(dataDirectory):
    data = loadCodeEvaluations(str(dataDirectory), l=6, m=6, errorRange=CANONICAL_ERROR_RANGE)
    row = data.samples.argmax()                # the codeA row
    np.testing.assert_array_equal(
        data.bits[row],
        np.array([1,0,1,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,1,0,0,0,0], dtype=np.int8))


def test_rewardMatchesTrapezoidReconstruction():
    counts = np.array([[0, 10, 20, 30, 40]])
    samples = np.array([50])
    ber = counts[0] / 50.0
    l,m = 6,6
    expected = np.trapezoid(1.0 - ber, CANONICAL_ERROR_RANGE)
    reward = calculateRewardFromSamples(counts, samples, CANONICAL_ERROR_RANGE, l ,m, rewardEngineering=False)
    assert reward.shape == (1,)
    np.testing.assert_allclose(reward[0], expected, rtol=1e-12)


def test_splitIsDisjointAndComplete(dataDirectory):
    data = loadCodeEvaluations(str(dataDirectory), l=6, m=6, errorRange=CANONICAL_ERROR_RANGE)
    train, val, test = splitData(data, fractions=(0.5, 0.5, 0.0), seed=0)
    assert train.bits.shape[0] + val.bits.shape[0] + test.bits.shape[0] == 2
    assert train.l == 6 and train.m == 6


def test_toTensorsShapesAndDtypes(dataDirectory):
    data = loadCodeEvaluations(str(dataDirectory), l=6, m=6, errorRange=CANONICAL_ERROR_RANGE)
    bits, counts, samples, k = toTensors(data)
    assert bits.dtype == torch.float32 and bits.shape == (2, 24)
    assert counts.shape == (2, 5) and samples.shape == (2,) and k.shape == (2,)


def test_generateProbesWritesLoadableRecords(tmp_path):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "surrogate"))
    from generateProbeEvaluations import generateProbes
    evaluated = generateProbes(l=6, m=6, budget=2, outputDirectory=str(tmp_path),
                               seed=7, numberOfSamples=5)
    assert evaluated >= 1
    data = loadCodeEvaluations(str(tmp_path), l=6, m=6, errorRange=CANONICAL_ERROR_RANGE)
    assert data.bits.shape[1] == 24
    assert data.samples.min() >= 5
