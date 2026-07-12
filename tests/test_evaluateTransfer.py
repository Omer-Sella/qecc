import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "surrogate"))
from evaluateTransfer import (topKOverlap, noiseFloorNll, binomialNllOfCurve,
                              evaluateOnData, appendReport)


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
    for key in ("nll", "noiseFloor", "rewardMae", "kMae", "spearman", "kendall", "topK50"):
        assert key in metrics and np.isfinite(metrics[key])


def test_appendReportWritesMetricsBlock(tmp_path):
    metrics = {"nll": 12.3456, "noiseFloor": 11.0, "rewardMae": 0.00123, "kMae": 1.5,
               "spearman": 0.42, "kendall": 0.31, "topK50": 0.6, "numberOfCodes": 200}
    reportPath = os.path.join(str(tmp_path), "nested", "report.md")  # dir does not exist yet
    text = appendReport(reportPath, "run title", "held-out 10% of /data", metrics)
    assert os.path.exists(reportPath)
    with open(reportPath) as fid:
        contents = fid.read()
    assert "## run title" in contents
    assert "Binomial NLL: **12.3456** (noise floor 11.0000)" in contents
    assert "k MAE: **1.50**" in contents
    assert "Spearman: **0.420**" in contents
    assert contents == text
    # a second call appends rather than overwrites
    appendReport(reportPath, "second run", "more data", metrics)
    with open(reportPath) as fid:
        assert fid.read().count("## ") == 2


def test_evaluateOnDataConstantRewardEmitsNoWarning():
    from qecc.codeEvaluationDataset import CodeEvaluationData
    from qecc.codeSurrogate import CodeCurvePredictor
    rng = np.random.default_rng(1)
    n = 12
    data = CodeEvaluationData(
        bits=rng.integers(0, 2, size=(n, 24)).astype(np.int8),
        counts=np.tile(np.array([0, 10, 20, 30, 40], dtype=np.int64), (n, 1)),
        samples=np.full(n, 50, dtype=np.int64),
        k=np.full(n, 8, dtype=np.int64), l=6, m=6)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        metrics = evaluateOnData(CodeCurvePredictor(dModel=32, nHead=2, numLayers=1,
                                                     dimFeedforward=64), data)
    assert np.isnan(metrics["spearman"])
    assert np.isnan(metrics["kendall"])
    for key in ("nll", "noiseFloor", "rewardMae", "kMae", "topK50"):
        assert np.isfinite(metrics[key])
