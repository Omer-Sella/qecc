import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "surrogate"))
from evaluateTransfer import topKOverlap, noiseFloorNll, binomialNllOfCurve, evaluateOnData


def test_topKOverlapPerfectAndDisjoint():
    values = np.arange(100, dtype=float)
    assert topKOverlap(values, values, 10) == 1.0
    assert topKOverlap(values, -values, 10) == 0.0


def test_noiseFloorIsBelowAnyOtherCurve():
    counts = np.array([[0, 10, 25, 40, 50]], dtype=float)
    samples = np.array([50.0])
    floor = noiseFloorNll(counts, samples)
    worse = binomialNllOfCurve(np.full((1, 5), 0.5), counts, samples)
    assert floor < worse


def test_evaluateOnDataReturnsAllMetrics():
    from qecc.codeEvaluationDataset import CodeEvaluationData
    from qecc.codeSurrogate import CodeCurvePredictor
    rng = np.random.default_rng(0)
    n = 60
    data = CodeEvaluationData(
        bits=rng.integers(0, 2, size=(n, 24)).astype(np.int8),
        counts=rng.integers(0, 51, size=(n, 5)).astype(np.int64),
        samples=np.full(n, 50, dtype=np.int64),
        k=np.full(n, 8, dtype=np.int64), l=6, m=6)
    metrics = evaluateOnData(CodeCurvePredictor(dModel=32, nHead=2, numLayers=1,
                                                dimFeedforward=64), data)
    for key in ("nll", "noiseFloor", "rewardMae", "spearman", "kendall", "topK50"):
        assert key in metrics and np.isfinite(metrics[key])
