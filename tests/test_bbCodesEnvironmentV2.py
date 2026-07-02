import numpy as np
import pytest
import qecc
import gymnasium as gym

NEGATIVE_REWARD = -1
TEST_ERROR_RANGE = np.linspace(10**-4, 10**-1, 10)

def _fake_decoder(Hx, Hz, errorRange, seed=None):
    n = len(errorRange)
    return np.zeros(n), np.zeros(n)


def _make_v2_env_with_fake_decoder(l=6, m=6, max_ax=5, max_ay=5, max_bx=5, max_by=5,
                 minimumNumberOfLogicalQubits=6):
    from qecc.bb_gym_v2 import bicycleBivariateCodeEnvironmentV2
    return bicycleBivariateCodeEnvironmentV2(
        l=l, m=m,
        max_ax=max_ax, max_ay=max_ay, max_bx=max_bx, max_by=max_by,
        evaluationDecoderFunction=_fake_decoder,
        errorRange=TEST_ERROR_RANGE,
        minimumNumberOfLogicalQubits=minimumNumberOfLogicalQubits,
    )

def _make_v2_env_with_dualBinaryBPOSDDecoder(l=6, m=6, max_ax=5, max_ay=5, max_bx=5, max_by=5,
                 minimumNumberOfLogicalQubits=6):
    from qecc.bb_gym_v2 import bicycleBivariateCodeEnvironmentV2
    from qecc.bb_gym import exampleDecoderFunction2
    return bicycleBivariateCodeEnvironmentV2(
        l=l, m=m,
        max_ax=max_ax, max_ay=max_ay, max_bx=max_bx, max_by=max_by,
        evaluationDecoderFunction=exampleDecoderFunction2,
        errorRange=TEST_ERROR_RANGE,
        minimumNumberOfLogicalQubits=minimumNumberOfLogicalQubits,
    )


def test_bbcodeV1IsRegistered():
    import qecc #noqa
    allEnvs = gym.envs.registry.keys()
    assert "qecc/bbcode-bitflip-v0" in allEnvs


def test_actionSpaceShape():
    env = _make_v2_env_with_dualBinaryBPOSDDecoder(max_ax=5, max_ay=5, max_bx=5, max_by=5)
    assert list(env.action_space.nvec) == [6, 6, 6, 6]  # max_p + 1 each


def test_observationSpaceSize():
    env = _make_v2_env_with_dualBinaryBPOSDDecoder(l=6, m=6, max_ax=5, max_ay=5, max_bx=5, max_by=5)
    expected = 2 * (6 * 6) ** 2 + 5 + 5 + 5 + 5  # = 2612
    assert env.flatObservationSize == expected
    assert env.observation_space.shape == (expected,)


def test_resetReturnsCorrectShapes():
    env = _make_v2_env_with_dualBinaryBPOSDDecoder(l=6, m=6, max_ax=5, max_ay=5, max_bx=5, max_by=5)
    obs, info = env.reset()
    assert obs.shape == (env.flatObservationSize,)
    assert isinstance(info, dict)


def test_resetZerosPolynomials():
    env = _make_v2_env_with_dualBinaryBPOSDDecoder(l=6, m=6, max_ax=5, max_ay=5, max_bx=5, max_by=5)
    env.reset()
    assert np.all(env.aX == 0)
    assert np.all(env.aY == 0)
    assert np.all(env.bX == 0)
    assert np.all(env.bY == 0)


def test_bitFlipChangesPolynomial():
    env = _make_v2_env_with_dualBinaryBPOSDDecoder(l=6, m=6, max_ax=5, max_ay=5, max_bx=5, max_by=5)
    env.reset()
    # flip aX[2]; no-op on aY (5), bX (5), bY (5)
    env.step(np.array([2, 5, 5, 5]))
    assert env.aX[2] == 1
    assert np.sum(env.aX) == 1
    assert np.all(env.aY == 0)
    assert np.all(env.bX == 0)
    assert np.all(env.bY == 0)


def test_doubleFlipRestores():
    env = _make_v2_env_with_dualBinaryBPOSDDecoder(l=6, m=6, max_ax=5, max_ay=5, max_bx=5, max_by=5)
    env.reset()
    env.step(np.array([2, 5, 5, 5]))
    env.step(np.array([2, 5, 5, 5]))
    assert env.aX[2] == 0


def test_noOpLeavesAllPolynomialsUnchanged():
    env = _make_v2_env_with_dualBinaryBPOSDDecoder(l=6, m=6, max_ax=5, max_ay=5, max_bx=5, max_by=5)
    env.reset()
    env.step(np.array([5, 5, 5, 5]))  # all no-ops
    assert np.all(env.aX == 0)
    assert np.all(env.aY == 0)
    assert np.all(env.bX == 0)
    assert np.all(env.bY == 0)


def test_stepReturnSignature():
    env = _make_v2_env_with_dualBinaryBPOSDDecoder(l=6, m=6, max_ax=5, max_ay=5, max_bx=5, max_by=5)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(np.array([2, 1, 3, 4]))
    assert obs.shape == (env.flatObservationSize,)
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert isinstance(info, dict)


def test_observationIncludesPolynomialCoefficients():
    env = _make_v2_env_with_dualBinaryBPOSDDecoder(l=6, m=6, max_ax=5, max_ay=5, max_bx=5, max_by=5)
    env.reset()
    obs, *_ = env.step(np.array([2, 5, 5, 5]))  # flip aX[2]
    # Polynomial part starts at index 2*(6*6)^2 = 2592
    poly_start = 2 * (6 * 6) ** 2
    aX_obs = obs[poly_start: poly_start + 5]
    assert aX_obs[2] == 1
    assert np.sum(aX_obs) == 1


def test_stepReturnsNegativeRewardWhenDimensionTooLow():
    # Use an impossibly high threshold so no code can satisfy it
    env = _make_v2_env_with_dualBinaryBPOSDDecoder(l=6, m=6, max_ax=5, max_ay=5, max_bx=5, max_by=5,minimumNumberOfLogicalQubits=10000)
    env.reset()
    _, reward, *_ = env.step(np.array([2, 1, 3, 4]))
    assert reward == NEGATIVE_REWARD


def test_stepCallsDecoderForValidCode():
    # IBM [[72,12,6]]: l=6,m=6, aX=[3], aY=[1,2], bX=[1,2], bY=[3]
    # Build this code by sequential single-bit flips; require k>=12
    env = _make_v2_env_with_dualBinaryBPOSDDecoder(l=6, m=6, max_ax=5, max_ay=5, max_bx=5, max_by=5,
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
    # _fake_decoder returns zeros; reward != NEGATIVE_REWARD confirms decoder was called
    assert reward != NEGATIVE_REWARD


def test_v2EnvPassesGymSpecCheck():
    import qecc
    torchrl_gym = pytest.importorskip("torchrl.envs.libs.gym", reason="torchrl not installed")
    torchrl_utils = pytest.importorskip("torchrl.envs.utils", reason="torchrl not installed")
    GymEnv = torchrl_gym.GymEnv
    check_env_specs = torchrl_utils.check_env_specs
    from qecc.bb_gym import exampleDecoderFunction2
    base_env = GymEnv(
        "qecc/bbcode-bitflip-v0",
        device="cpu",
        l=6, m=6,
        max_ax=5, max_ay=5, max_bx=5, max_by=5,
        evaluationDecoderFunction=exampleDecoderFunction2,
        errorRange=TEST_ERROR_RANGE,
        minimumNumberOfLogicalQubits=6,
    )
    check_env_specs(base_env)


# ---------------------------------------------------------------------------
# IBM Table 1 codes — positive reward checks (ascending errorRange required)
# A(x,y) = sum_{i in aX} x^i  +  sum_{j in aY} y^j
# B(x,y) = sum_{i in bX} x^i  +  sum_{j in bY} y^j
# ---------------------------------------------------------------------------

def _build_IBM_code_v2(env, aX_idx, aY_idx, bX_idx, bY_idx, max_ax, max_ay, max_bx, max_by):
    """Build a code by sequential single-bit flips."""
    obs, reward = None, None
    for idx in aX_idx:
        obs, reward, *_ = env.step(np.array([idx, max_ay, max_bx, max_by]))
    for idx in aY_idx:
        obs, reward, *_ = env.step(np.array([max_ax, idx, max_bx, max_by]))
    for idx in bX_idx:
        obs, reward, *_ = env.step(np.array([max_ax, max_ay, idx, max_by]))
    for idx in bY_idx:
        obs, reward, *_ = env.step(np.array([max_ax, max_ay, max_bx, idx]))
    return reward


def test_v2_IBM_72_12_6_positiveReward():
    """[[72, 12, 6]]: l=6, m=6, A=x³+y+y², B=y³+x+x²"""
    env = _make_v2_env_with_dualBinaryBPOSDDecoder(l=6, m=6, max_ax=5, max_ay=5, max_bx=5, max_by=5,
                       minimumNumberOfLogicalQubits=1)
    env.reset()
    reward = _build_IBM_code_v2(env, [3], [1, 2], [1, 2], [3], 5, 5, 5, 5)
    assert reward > 0.029


def test_v2_IBM_90_8_10_positiveReward():
    """[[90, 8, 10]]: l=15, m=3, A=x⁹+y+y², B=1+x²+x⁷"""
    env = _make_v2_env_with_dualBinaryBPOSDDecoder(l=15, m=3, max_ax=10, max_ay=3, max_bx=8, max_by=1,
                       minimumNumberOfLogicalQubits=1)
    env.reset()
    reward = _build_IBM_code_v2(env, [9], [1, 2], [0, 2, 7], [], 10, 3, 8, 1)
    assert reward > 0.035


def test_v2_IBM_108_8_10_positiveReward():
    """[[108, 8, 10]]: l=9, m=6, A=x³+y+y², B=y³+x+x²"""
    env = _make_v2_env_with_dualBinaryBPOSDDecoder(l=9, m=6, max_ax=4, max_ay=3, max_bx=3, max_by=4,
                       minimumNumberOfLogicalQubits=1)
    env.reset()
    reward = _build_IBM_code_v2(env, [3], [1, 2], [1, 2], [3], 4, 3, 3, 4)
    assert reward > 0.035


def test_v2_IBM_144_12_12_positiveReward():
    """[[144, 12, 12]]: l=12, m=6, A=x³+y+y², B=y³+x+x²"""
    env = _make_v2_env_with_dualBinaryBPOSDDecoder(l=12, m=6, max_ax=4, max_ay=3, max_bx=3, max_by=4,
                       minimumNumberOfLogicalQubits=1)
    env.reset()
    reward = _build_IBM_code_v2(env, [3], [1, 2], [1, 2], [3], 4, 3, 3, 4)
    assert reward > 0.03


def test_v2_IBM_288_12_18_positiveReward():
    """[[288, 12, 18]]: l=12, m=12, A=x³+y²+y⁷, B=y³+x+x²"""
    env = _make_v2_env_with_dualBinaryBPOSDDecoder(l=12, m=12, max_ax=4, max_ay=8, max_bx=3, max_by=4,
                       minimumNumberOfLogicalQubits=1)
    env.reset()
    reward = _build_IBM_code_v2(env, [3], [2, 7], [1, 2], [3], 4, 8, 3, 4)
    assert reward > 0.03




if __name__ == "__main__":
    test_actionSpaceShape()
    test_bbcodeV1IsRegistered()
    test_bitFlipChangesPolynomial()
    test_doubleFlipRestores()
    test_v2EnvPassesGymSpecCheck()
    test_v2_IBM_90_8_10_positiveReward()
    test_v2_IBM_72_12_6_positiveReward()
    test_v2_IBM_144_12_12_positiveReward()
    test_v2_IBM_108_8_10_positiveReward()
    test_stepReturnsNegativeRewardWhenDimensionTooLow()
    test_noOpLeavesAllPolynomialsUnchanged()
    
    
    
