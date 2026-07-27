import numpy as np
import pytest
import torch
from qecc.codeSurrogate import (CodeCurvePredictor, binomialCurveLoss,
                                kPredictionLoss)

from qecc.utils import calculateRewardFromSamples, CANONICAL_ERROR_RANGE


def makeBatch(l, m, batchSize=4, seed=0):
    generator = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 2, (batchSize, 2 * l + 2 * m), generator=generator).float()
    k = torch.full((batchSize,), 8.0)
    return bits, k


def test_forwardShapesAndCurveRange():
    model = CodeCurvePredictor()
    bits, _k = makeBatch(6, 6)
    curveLogits, kLogPrediction = model(bits, 6, 6)
    assert curveLogits.shape == (4, 5)
    assert kLogPrediction.shape == (4,)
    curve = model.predictCurve(bits, 6, 6)
    assert torch.all(curve >= 0) and torch.all(curve <= 1)


def test_sameWeightsRunAtAnySize():
    source = CodeCurvePredictor()
    target = CodeCurvePredictor()
    target.load_state_dict(source.state_dict(), strict=True)
    bits12, _k12 = makeBatch(12, 12)
    curveLogits, kLogPrediction = target(bits12, 12, 12)
    assert curveLogits.shape == (4, 5) and kLogPrediction.shape == (4,)


def test_kPredictionLossZeroAtExactPrediction():
    k = torch.tensor([6.0, 8.0, 72.0])
    assert kPredictionLoss(torch.log1p(k), k).item() == 0.0
    assert kPredictionLoss(torch.log1p(k) + 1.0, k).item() > 0.0


def test_rewardCollapseMatchesNumpyTrapezoid():
    curve = torch.tensor([[0.0, 0.2, 0.4, 0.6, 0.8]])
    expected = np.trapezoid(1.0 - curve.numpy()[0], CANONICAL_ERROR_RANGE)
    reward = calculateRewardFromSamples(curve, numberOfSamples=1, errorRange=CANONICAL_ERROR_RANGE, l = 6, m = 6, rewardEngineering=False)
    np.testing.assert_allclose(reward, torch.tensor([expected], dtype=torch.float32),
                               rtol=1e-6, atol=1e-8)


def test_binomialLossIsMinimizedAtEmpiricalRate():
    counts = torch.tensor([[0.0, 10.0, 25.0, 40.0, 50.0]])
    samples = torch.tensor([50.0])
    empirical = counts / samples.unsqueeze(-1)
    epsilon = 1e-4
    logitsAtEmpirical = torch.logit(empirical.clamp(epsilon, 1 - epsilon))
    lossAtEmpirical = binomialCurveLoss(logitsAtEmpirical, counts, samples)
    lossElsewhere = binomialCurveLoss(logitsAtEmpirical + 0.5, counts, samples)
    assert lossAtEmpirical < lossElsewhere


def test_singleBatchOverfitRecoversCurve():
    torch.manual_seed(0)
    model = CodeCurvePredictor(dModel=32, nHead=2, numLayers=1, dimFeedforward=64)
    bits, _k = makeBatch(6, 6, batchSize=2)
    counts = torch.tensor([[0.0, 10.0, 25.0, 40.0, 50.0],
                           [0.0,  2.0,  5.0, 20.0, 45.0]])
    samples = torch.tensor([50.0, 50.0])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()  # dropout on during training, matching the real pipeline
    for _ in range(1500):
        optimizer.zero_grad()
        curveLogits, _kLogPrediction = model(bits, 6, 6)
        loss = binomialCurveLoss(curveLogits, counts, samples)
        loss.backward()
        optimizer.step()
    model.eval()  # dropout OFF for evaluation: read the learned mapping, not a randomly-masked one
    curve = model.predictCurve(bits, 6, 6).detach()
    torch.testing.assert_close(curve, counts / 50.0, atol=0.05, rtol=0.0)


def test_trainModelSmoke(tmp_path):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "surrogate"))
    from trainSurrogate import trainModel
    from qecc.codeEvaluationDataset import CodeEvaluationData
    import numpy as np
    rng = np.random.default_rng(0)
    n = 64
    data = CodeEvaluationData(
        bits=rng.integers(0, 2, size=(n, 24)).astype(np.int8),
        counts=rng.integers(0, 51, size=(n, 5)).astype(np.int64),
        samples=np.full(n, 50, dtype=np.int64),
        k=np.full(n, 8, dtype=np.int64), l=6, m=6)
    model, history, testData = trainModel(data, epochs=2, batchSize=32, lr=1e-3, seed=0,
                                           dModel=32, nHead=2, numLayers=1, dimFeedforward=64)
    assert len(history["trainLoss"]) == 2
    assert all(np.isfinite(v) for v in history["trainLoss"] + history["valLoss"])
    assert testData.bits.shape[0] > 0  # held-out test split returned for reporting


def test_trainModelRaisesOnEmptySplit(tmp_path):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "surrogate"))
    from trainSurrogate import trainModel
    from qecc.codeEvaluationDataset import CodeEvaluationData
    import numpy as np
    rng = np.random.default_rng(0)
    n = 5
    data = CodeEvaluationData(
        bits=rng.integers(0, 2, size=(n, 24)).astype(np.int8),
        counts=rng.integers(0, 51, size=(n, 5)).astype(np.int64),
        samples=np.full(n, 50, dtype=np.int64),
        k=np.full(n, 8, dtype=np.int64), l=6, m=6)
    with pytest.raises(ValueError):
        trainModel(data, epochs=1, batchSize=32, lr=1e-3, seed=0,
                   dModel=32, nHead=2, numLayers=1, dimFeedforward=64)
