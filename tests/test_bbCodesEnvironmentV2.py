import numpy as np
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
