import numpy as np
import torch
from qecc.codeSurrogate import CodeCurvePredictor, binomialCurveLoss, rewardFromCurve
from qecc.codeEvaluationDataset import CANONICAL_ERROR_RANGE


def makeBatch(l, m, batchSize=4, seed=0):
    generator = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 2, (batchSize, 2 * l + 2 * m), generator=generator).float()
    k = torch.full((batchSize,), 8.0)
    return bits, k


def test_forwardShapesAndCurveRange():
    model = CodeCurvePredictor()
    bits, k = makeBatch(6, 6)
    logits = model(bits, 6, 6, k)
    assert logits.shape == (4, 5)
    curve = model.predictCurve(bits, 6, 6, k)
    assert torch.all(curve >= 0) and torch.all(curve <= 1)


def test_sameWeightsRunAtAnySize():
    source = CodeCurvePredictor()
    target = CodeCurvePredictor()
    target.load_state_dict(source.state_dict(), strict=True)
    bits12, k12 = makeBatch(12, 12)
    assert target(bits12, 12, 12, k12).shape == (4, 5)


def test_rewardCollapseMatchesNumpyTrapezoid():
    curve = torch.tensor([[0.0, 0.2, 0.4, 0.6, 0.8]])
    expected = np.trapezoid(1.0 - curve.numpy()[0], CANONICAL_ERROR_RANGE)
    reward = rewardFromCurve(curve, CANONICAL_ERROR_RANGE)
    torch.testing.assert_close(reward, torch.tensor([expected], dtype=torch.float32),
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
    bits, k = makeBatch(6, 6, batchSize=2)
    counts = torch.tensor([[0.0, 10.0, 25.0, 40.0, 50.0],
                           [0.0,  2.0,  5.0, 20.0, 45.0]])
    samples = torch.tensor([50.0, 50.0])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()  # dropout on during training, matching the real pipeline
    for _ in range(1500):
        optimizer.zero_grad()
        loss = binomialCurveLoss(model(bits, 6, 6, k), counts, samples)
        loss.backward()
        optimizer.step()
    model.eval()  # dropout OFF for evaluation: read the learned mapping, not a randomly-masked one
    curve = model.predictCurve(bits, 6, 6, k).detach()
    torch.testing.assert_close(curve, counts / 50.0, atol=0.05, rtol=0.0)
