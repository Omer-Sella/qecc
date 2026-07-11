# tests/test_codeEvaluationDataset.py
import json
import numpy as np
import pytest
import torch
from qecc.codeEvaluationDataset import (
    CANONICAL_ERROR_RANGE, CodeEvaluationData,
    loadCodeEvaluations, rewardFromCounts, splitData, toTensors,
)


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
    data = loadCodeEvaluations(str(dataDirectory), l=6, m=6)
    assert isinstance(data, CodeEvaluationData)
    assert data.bits.shape == (2, 24)          # 2 unique codes; bad-grid record dropped
    assert data.counts.shape == (2, 5)
    duplicatedRow = data.samples.argmax()
    assert data.samples[duplicatedRow] == 100  # 50 + 50 summed
    # combined counts of the duplicated code: logical+failures summed across both records
    expected = np.array([0, 22, 40, 60, 90])   # (0+0+0+0, 10+12, 20+18+2, 30+30, 40+5+40+5)
    np.testing.assert_array_equal(data.counts[duplicatedRow], expected)


def test_bitOrderIsAxAyBxBy(dataDirectory):
    data = loadCodeEvaluations(str(dataDirectory), l=6, m=6)
    row = data.samples.argmax()                # the codeA row
    np.testing.assert_array_equal(
        data.bits[row],
        np.array([1,0,1,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,1,0,0,0,0], dtype=np.int8))


def test_rewardMatchesTrapezoidReconstruction():
    counts = np.array([[0, 10, 20, 30, 40]])
    samples = np.array([50])
    ber = counts[0] / 50.0
    expected = np.trapezoid(1.0 - ber, CANONICAL_ERROR_RANGE)
    reward = rewardFromCounts(counts, samples, CANONICAL_ERROR_RANGE)
    assert reward.shape == (1,)
    np.testing.assert_allclose(reward[0], expected, rtol=1e-12)


def test_splitIsDisjointAndComplete(dataDirectory):
    data = loadCodeEvaluations(str(dataDirectory), l=6, m=6)
    train, val, test = splitData(data, fractions=(0.5, 0.5, 0.0), seed=0)
    assert train.bits.shape[0] + val.bits.shape[0] + test.bits.shape[0] == 2
    assert train.l == 6 and train.m == 6


def test_toTensorsShapesAndDtypes(dataDirectory):
    data = loadCodeEvaluations(str(dataDirectory), l=6, m=6)
    bits, counts, samples, k = toTensors(data)
    assert bits.dtype == torch.float32 and bits.shape == (2, 24)
    assert counts.shape == (2, 5) and samples.shape == (2,) and k.shape == (2,)


def test_overLengthRecordIsFolded(tmp_path):
    # Well-formed length-6 code: aX bit at index 2.
    good = makeRecord(aX=[0, 0, 1, 0, 0, 0], aY=[0, 1, 0, 0, 0, 0],
                      bX=[0, 0, 0, 1, 0, 0], bY=[0, 0, 0, 0, 1, 0],
                      logical=[0, 5, 10, 20, 30], failures=[0, 0, 0, 0, 0], seed=1)
    # Over-length length-36 record that folds to the SAME code: aX bit at index 8
    # (8 % 6 == 2), and aY/bX/bY over-length with a single set bit that reduces to
    # the same residues (index 7 -> 1 for aY, 3 -> 3 for bX, 10 -> 4 for bY).
    overLengthAX = [0] * 36
    overLengthAX[8] = 1
    overLengthAY = [0] * 36
    overLengthAY[7] = 1
    overLengthBX = [0] * 36
    overLengthBX[3] = 1
    overLengthBY = [0] * 36
    overLengthBY[10] = 1
    overLength = makeRecord(aX=overLengthAX, aY=overLengthAY,
                            bX=overLengthBX, bY=overLengthBY,
                            logical=[0, 3, 6, 12, 18], failures=[0, 0, 0, 0, 0], seed=2)
    overLength["l"] = 6
    overLength["m"] = 6
    subdir = tmp_path / "2026-07-11_00-00-00"
    subdir.mkdir()
    with open(subdir / "codeEvaluations_9999.jsonl", "w") as f:
        for r in (good, overLength):
            f.write(json.dumps(r) + "\n")
    data = loadCodeEvaluations(str(tmp_path), l=6, m=6)
    # (a) The two records fold to the same code -> ONE row.
    assert data.bits.shape == (1, 24)
    np.testing.assert_array_equal(
        data.bits[0],
        np.array([0,0,1,0,0,0, 0,1,0,0,0,0, 0,0,0,1,0,0, 0,0,0,0,1,0], dtype=np.int8))
    # (b) counts and samples are summed across both records.
    np.testing.assert_array_equal(data.counts[0], np.array([0, 8, 16, 32, 48]))
    assert data.samples[0] == 100


def test_foldIsXorNotOr(tmp_path):
    # A single length-12 aX with bits set at BOTH index 2 and index 8 (2 == 8 mod 6):
    # XOR-folding cancels them to 0 at residue 2 (OR-folding would leave a 1 there).
    overLengthAX = [0] * 12
    overLengthAX[2] = 1
    overLengthAX[8] = 1
    record = makeRecord(aX=overLengthAX, aY=[0, 0, 0, 0, 0, 0],
                        bX=[0, 0, 0, 0, 0, 0], bY=[0, 0, 0, 0, 0, 0],
                        logical=[0, 1, 2, 3, 4], failures=[0, 0, 0, 0, 0], seed=1)
    record["l"] = 6
    record["m"] = 6
    subdir = tmp_path / "2026-07-11_00-00-00"
    subdir.mkdir()
    with open(subdir / "codeEvaluations_8888.jsonl", "w") as f:
        f.write(json.dumps(record) + "\n")
    data = loadCodeEvaluations(str(tmp_path), l=6, m=6)
    # Residue 2 of the folded aX block (first 6 bits) must be 0 (parity cancellation).
    assert data.bits[0][2] == 0
    # Whole code folds to all zeros.
    np.testing.assert_array_equal(data.bits[0], np.zeros(24, dtype=np.int8))


def test_generateProbesWritesLoadableRecords(tmp_path):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "surrogate"))
    from generateProbeEvaluations import generateProbes
    evaluated = generateProbes(l=6, m=6, budget=2, outputDirectory=str(tmp_path),
                               seed=7, numberOfSamples=5)
    assert evaluated >= 1
    data = loadCodeEvaluations(str(tmp_path), l=6, m=6)
    assert data.bits.shape[1] == 24
    assert data.samples.min() >= 5
