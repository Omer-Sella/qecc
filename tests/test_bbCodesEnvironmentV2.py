import numpy as np
import pytest
import qecc
import gymnasium as gym

NEGATIVE_REWARD = -1


def _fast_decoder(Hx, Hz, errorRange, seed=None):
    n = len(errorRange)
    return np.zeros(n), np.zeros(n)


def _make_v2_env(l=6, m=6, max_ax=5, max_ay=5, max_bx=5, max_by=5,
                 minimumNumberOfLogicalQubits=6):
    from qecc.bb_gym_v2 import bicycleBivariateCodeEnvironmentV2
    return bicycleBivariateCodeEnvironmentV2(
        l=l, m=m,
        max_ax=max_ax, max_ay=max_ay, max_bx=max_bx, max_by=max_by,
        evaluationDecoderFunction=_fast_decoder,
        errorRange=[0.01, 0.001],
        minimumNumberOfLogicalQubits=minimumNumberOfLogicalQubits,
    )


def test_bbcodeV1IsRegistered():
    allEnvs = gym.envs.registry.keys()
    assert "qecc/bbcode-v1" in allEnvs


def test_actionSpaceShape():
    env = _make_v2_env(max_ax=5, max_ay=5, max_bx=5, max_by=5)
    assert list(env.action_space.nvec) == [6, 6, 6, 6]  # max_p + 1 each


def test_observationSpaceSize():
    env = _make_v2_env(l=6, m=6, max_ax=5, max_ay=5, max_bx=5, max_by=5)
    expected = 2 * (6 * 6) ** 2 + 5 + 5 + 5 + 5  # = 2612
    assert env.flatObservationSize == expected
    assert env.observation_space.shape == (expected,)


def test_resetReturnsCorrectShapes():
    env = _make_v2_env()
    obs, info = env.reset()
    assert obs.shape == (env.flatObservationSize,)
    assert isinstance(info, dict)


def test_resetZerosPolynomials():
    env = _make_v2_env()
    env.reset()
    assert np.all(env.aX == 0)
    assert np.all(env.aY == 0)
    assert np.all(env.bX == 0)
    assert np.all(env.bY == 0)


def test_bitFlipChangesPolynomial():
    env = _make_v2_env()
    env.reset()
    # flip aX[2]; no-op on aY (5), bX (5), bY (5)
    env.step(np.array([2, 5, 5, 5]))
    assert env.aX[2] == 1
    assert np.sum(env.aX) == 1
    assert np.all(env.aY == 0)
    assert np.all(env.bX == 0)
    assert np.all(env.bY == 0)


def test_doubleFlipRestores():
    env = _make_v2_env()
    env.reset()
    env.step(np.array([2, 5, 5, 5]))
    env.step(np.array([2, 5, 5, 5]))
    assert env.aX[2] == 0


def test_noOpLeavesAllPolynomialsUnchanged():
    env = _make_v2_env(max_ax=5, max_ay=5, max_bx=5, max_by=5)
    env.reset()
    env.step(np.array([5, 5, 5, 5]))  # all no-ops
    assert np.all(env.aX == 0)
    assert np.all(env.aY == 0)
    assert np.all(env.bX == 0)
    assert np.all(env.bY == 0)


def test_stepReturnSignature():
    env = _make_v2_env()
    env.reset()
    obs, reward, terminated, truncated, info = env.step(np.array([2, 1, 3, 4]))
    assert obs.shape == (env.flatObservationSize,)
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert isinstance(info, dict)


def test_observationIncludesPolynomialCoefficients():
    env = _make_v2_env(l=6, m=6, max_ax=5, max_ay=5, max_bx=5, max_by=5)
    env.reset()
    obs, *_ = env.step(np.array([2, 5, 5, 5]))  # flip aX[2]
    # Polynomial part starts at index 2*(6*6)^2 = 2592
    poly_start = 2 * (6 * 6) ** 2
    aX_obs = obs[poly_start: poly_start + 5]
    assert aX_obs[2] == 1
    assert np.sum(aX_obs) == 1


def test_stepReturnsNegativeRewardWhenDimensionTooLow():
    # Use an impossibly high threshold so no code can satisfy it
    env = _make_v2_env(minimumNumberOfLogicalQubits=10000)
    env.reset()
    _, reward, *_ = env.step(np.array([2, 1, 3, 4]))
    assert reward == NEGATIVE_REWARD


def test_stepCallsDecoderForValidCode():
    # IBM [[72,12,6]]: l=6,m=6, aX=[3], aY=[1,2], bX=[1,2], bY=[3]
    # Build this code by sequential single-bit flips; require k>=12
    env = _make_v2_env(l=6, m=6, max_ax=5, max_ay=5, max_bx=5, max_by=5,
                       minimumNumberOfLogicalQubits=12)
    env.reset()
    for action in [
        np.array([3, 5, 5, 5]),   # aX[3] = 1
        np.array([5, 1, 5, 5]),   # aY[1] = 1
        np.array([5, 2, 5, 5]),   # aY[2] = 1
        np.array([5, 5, 1, 5]),   # bX[1] = 1
        np.array([5, 5, 2, 5]),   # bX[2] = 1
        np.array([5, 5, 5, 3]),   # bY[3] = 1
    ]:
        _, reward, *_ = env.step(action)
    # _fast_decoder returns zeros; reward != NEGATIVE_REWARD confirms decoder was called
    assert reward != NEGATIVE_REWARD


def test_v2EnvPassesGymSpecCheck():
    import qecc
    torchrl_gym = pytest.importorskip("torchrl.envs.libs.gym", reason="torchrl not installed")
    torchrl_utils = pytest.importorskip("torchrl.envs.utils", reason="torchrl not installed")
    GymEnv = torchrl_gym.GymEnv
    check_env_specs = torchrl_utils.check_env_specs

    base_env = GymEnv(
        "qecc/bbcode-v1",
        device="cpu",
        l=6, m=6,
        max_ax=5, max_ay=5, max_bx=5, max_by=5,
        evaluationDecoderFunction=_fast_decoder,
        errorRange=[0.01, 0.001],
        minimumNumberOfLogicalQubits=6,
    )
    check_env_specs(base_env)
