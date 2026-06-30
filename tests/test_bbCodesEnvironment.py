"""
Table 1 from:
  "High-threshold and low-overhead fault-tolerant quantum memory"
  Bravyi, Cross, Gambetta, Nazarov, Rall, Yoder — Nature 2024 (arXiv:2308.07915)

Small examples of Bivariate Bicycle (BB) LDPC codes and their performance for the
circuit-based noise model.  All codes have weight-6 checks, thickness-2 Tanner graph,
and a depth-7 syndrome measurement circuit.  A code with parameters [[n, k, d]] requires
2n physical qubits in total and achieves the net encoding rate r = k/2n (we round r down
to the nearest inverse integer).  Circuit-level distance d_circ is the minimum number of
faulty operations in the syndrome measurement circuit required to generate an undetectable
logical error.  The pseudo-threshold p_0 is a solution of the break-even equation
p_L(p) = k·p, where p and p_L are the physical and logical error rates respectively.

  ┌────────────────┬───────────┬─────────┬────────┬───────────┬────────────┐
  │  [[n, k, d]]  │  Rate r   │ d_circ  │  p_0   │  p_L(p1)  │  p_L(p2)  │
  ├────────────────┼───────────┼─────────┼────────┼───────────┼────────────┤
  │ [[72,  12,  6]]│   1/12    │   ≤6    │ 0.0048 │  7×10⁻⁵  │  7×10⁻⁸  │
  │ [[90,   8, 10]]│   1/23    │   ≤8    │ 0.0053 │  5×10⁻⁶  │  4×10⁻¹⁰ │
  │ [[108,  8, 10]]│   1/27    │   ≤8    │ 0.0058 │  3×10⁻⁶  │  1×10⁻¹⁰ │
  │ [[144, 12, 12]]│   1/24    │   ≤10   │ 0.0065 │  2×10⁻⁷  │  8×10⁻¹³ │
  │ [[288, 12, 18]]│   1/48    │   ≤18   │ 0.0069 │  2×10⁻¹² │  1×10⁻²² │
  └────────────────┴───────────┴─────────┴────────┴───────────┴────────────┘
"""

import numpy as np
import qecc

TEST_ERROR_RANGE = np.linspace(0.0001,0.1,10)

def _make_action(l, m, aX_idx, aY_idx, bX_idx, bY_idx):
    """Build the flat MultiBinary action vector [aX, aY, bX, bY] from exponent lists."""
    
    aX = np.zeros(l * m, dtype=np.int8)
    aY = np.zeros(l * m, dtype=np.int8)
    bX = np.zeros(l * m, dtype=np.int8)
    bY = np.zeros(l * m, dtype=np.int8)
    for i in aX_idx:
        aX[i] = 1
    for i in aY_idx:
        aY[i] = 1
    for i in bX_idx:
        bX[i] = 1
    for i in bY_idx:
        bY[i] = 1
    return np.concatenate([aX, aY, bX, bY])


def test_bbCodesEnvIsWorking():

    from qecc.minSum import ldpcDecoderWrapper
    from qecc.utils import decoderEvaluator
    def decoderFunction(Hx,Hz,errorRange):
        numberOfSamples = 30
        logicalErrors, decoderFailures =  decoderEvaluator(decoderFunction = ldpcDecoderWrapper, dualBinary = True, Hx = Hx, Hz = Hz, errorRange = errorRange, decoderStoppingCriterion = 50, numberOfSamples = numberOfSamples)
        #return {key: value/numberOfSamples for key,value in logicalErrors.items()} , {key: value/numberOfSamples for key,value in decoderFailures.items()}
        return logicalErrors/numberOfSamples, decoderFailures/numberOfSamples
    # l = 6
    # m = 6
    # env = gym.make('qecc/bbcode-v0', l = l, m = m, evaluationDecoderFunction = decoderFunction, errorRange = [0.01, 0.001])
    # env.reset()
    # aX = np.zeros(l*m)
    # aY = np.zeros(l*m)
    # bX = np.zeros(l*m)
    # bY = np.zeros(l*m)
    # aX[3] = 1
    # aY[1]=1
    # aY[2] = 1
    # bX[1] = 1
    # bX[2] = 1
    # bY[3] = 1
    # action = np.hstack((np.hstack((aX,aY)), np.hstack((bX,bY))))
    # observation = env.step(action = action)
    # print(env.unwrapped.flatObservationSize)



def test_bbCodesEnvIsRegistered():
    import gymnasium as gym
    # Once qecc is imported, it's init file should have registered the environment. We can check that by checking the registry of gymnasium.
    import qecc
    allEnvs = gym.envs.registry.keys()
    assert "qecc/bbcode-v0" in allEnvs

def test_envPassesBasicChecks():
    import qecc
    import gymnasium as gym
    from qecc.bb_gym import exampleDecoderFunction
    import torch
    import numpy as np
    l = 6
    m = 6
    device = 'cpu'

    # Check the environment works with GymEnv
    from torchrl.envs.libs.gym import GymEnv
    from torchrl.envs.utils import check_env_specs
    base_env = GymEnv("qecc/bbcode-v0", device=device, l = 6, m = 6, evaluationDecoderFunction = exampleDecoderFunction, errorRange = TEST_ERROR_RANGE, minimumNumberOfLogicalQubits = 6)
    check_env_specs(base_env)


    # Check with gym.make
    env = gym.make('qecc/bbcode-v0', l = 6, m = 6, evaluationDecoderFunction = exampleDecoderFunction, errorRange = TEST_ERROR_RANGE)
    env.reset()
    print(env.action_space.shape)
    #print(env.unwrapped.flatObservationSize)
    aX = np.zeros(l*m)
    aY = np.zeros(l*m)
    bX = np.zeros(l*m)
    bY = np.zeros(l*m)
    print(env.action_space.shape)
    aX[3] = 1
    aY[1]=1
    aY[2] = 1
    bX[1] = 1
    bX[2] = 1
    bY[3] = 1
    action = np.concatenate([aX,aY,bX,bY])
    observation = env.step(action = action)
    pass


    def environmentFunction():
        return gym.make('qecc/bbcode-v0', l = 6, m = 6, evaluationDecoderFunction = exampleDecoderFunction, errorRange = TEST_ERROR_RANGE)

    env = environmentFunction()
    env.reset()
    action = torch.tensor([0., 1., 0., 1., 1., 0., 1., 0., 0., 1., 1., 0., 0., 0., 1., 1., 1., 1.,
        0., 1., 0., 1., 0., 1., 1., 0., 1., 1., 1., 0., 1., 1., 0., 1., 0., 0.,
        1., 1., 1., 0., 1., 0., 1., 0., 0., 1., 1., 1., 0., 1., 0., 1., 0., 0.,
        1., 0., 1., 1., 1., 1., 0., 0., 0., 1., 0., 0., 1., 1., 0., 1., 0., 0.,
        1., 1., 0., 1., 1., 0., 0., 0., 1., 1., 0., 1., 1., 0., 0., 1., 0., 1.,
        1., 1., 0., 0., 0., 1., 0., 1., 0., 1., 1., 0., 0., 0., 0., 1., 1., 1.,
        1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0.,
        1., 0., 0., 1., 0., 1., 1., 0., 1., 0., 0., 1., 0., 1., 0., 1., 1., 1.])
    action = action.numpy()
    next_o, r, d, _, info = env.step(action)
    pass


# ---------------------------------------------------------------------------
# IBM Table 1 codes — Table 3 polynomial parameters (arXiv:2308.07915)
# A(x,y) = sum_{i in aX} x^i  +  sum_{j in aY} y^j
# B(x,y) = sum_{i in bX} x^i  +  sum_{j in bY} y^j
# ---------------------------------------------------------------------------

def test_IBM_72_12_6():
    """[[72, 12, 6]]: l=6, m=6, A=x³+y+y², B=y³+x+x²"""
    import gymnasium as gym
    from qecc.bb_gym import exampleDecoderFunction2

    l, m = 6, 6
    env = gym.make(
        'qecc/bbcode-v0',
        l=l, m=m,
        evaluationDecoderFunction=exampleDecoderFunction2,
        errorRange=TEST_ERROR_RANGE,
        minimumNumberOfLogicalQubits=12,
    )
    env.reset()
    
    
    action = _make_action(l, m, aX_idx=[3], aY_idx=[1, 2], bX_idx=[1, 2], bY_idx=[3])
    _, reward, *_ = env.step(action)
    print(reward)
    # TODO: add assertions on reward / code parameters


def test_IBM_90_8_10():
    """[[90, 8, 10]]: l=15, m=3, A=x⁹+y+y², B=1+x²+x⁷"""
    import gymnasium as gym
    from qecc.bb_gym import exampleDecoderFunction

    l, m = 15, 3
    env = gym.make(
        'qecc/bbcode-v0',
        l=l, m=m,
        evaluationDecoderFunction=exampleDecoderFunction,
        errorRange=TEST_ERROR_RANGE,
        minimumNumberOfLogicalQubits=8,
    )
    env.reset()
    action = _make_action(l, m, aX_idx=[9], aY_idx=[1, 2], bX_idx=[0, 2, 7], bY_idx=[])
    _, reward, *_ = env.step(action)
    # TODO: add assertions on reward / code parameters


def test_IBM_108_8_10():
    """[[108, 8, 10]]: l=9, m=6, A=x³+y+y², B=y³+x+x²"""
    import gymnasium as gym
    from qecc.bb_gym import exampleDecoderFunction

    l, m = 9, 6
    env = gym.make(
        'qecc/bbcode-v0',
        l=l, m=m,
        evaluationDecoderFunction=exampleDecoderFunction,
        errorRange=TEST_ERROR_RANGE,
        minimumNumberOfLogicalQubits=8,
    )
    env.reset()
    action = _make_action(l, m, aX_idx=[3], aY_idx=[1, 2], bX_idx=[1, 2], bY_idx=[3])
    _, reward, *_ = env.step(action)
    # TODO: add assertions on reward / code parameters


def test_IBM_144_12_12():
    """[[144, 12, 12]]: l=12, m=6, A=x³+y+y², B=y³+x+x²"""
    import gymnasium as gym
    from qecc.bb_gym import exampleDecoderFunction

    l, m = 12, 6
    env = gym.make(
        'qecc/bbcode-v0',
        l=l, m=m,
        evaluationDecoderFunction=exampleDecoderFunction,
        errorRange=TEST_ERROR_RANGE,
        minimumNumberOfLogicalQubits=12,
    )
    env.reset()
    action = _make_action(l, m, aX_idx=[3], aY_idx=[1, 2], bX_idx=[1, 2], bY_idx=[3])
    _, reward, *_ = env.step(action)
    # TODO: add assertions on reward / code parameters


def test_IBM_288_12_18():
    """[[288, 12, 18]]: l=12, m=12, A=x³+y²+y⁷, B=y³+x+x²"""
    import gymnasium as gym
    from qecc.bb_gym import exampleDecoderFunction

    l, m = 12, 12
    env = gym.make(
        'qecc/bbcode-v0',
        l=l, m=m,
        evaluationDecoderFunction=exampleDecoderFunction,
        errorRange=TEST_ERROR_RANGE,
        minimumNumberOfLogicalQubits=12,
    )
    env.reset()
    action = _make_action(l, m, aX_idx=[3], aY_idx=[2, 7], bX_idx=[1, 2], bY_idx=[3])
    _, reward, *_ = env.step(action)
    # TODO: add assertions on reward / code parameters


# ---------------------------------------------------------------------------
# Positive reward checks using ascending errorRange and a fast decoder.
# These use minimumNumberOfLogicalQubits=1 so the decoder is always called.
# ---------------------------------------------------------------------------

def _fast_decoder(Hx, Hz, errorRange, seed=None):
    return np.zeros(len(errorRange)), np.zeros(len(errorRange))


def test_IBM_72_12_6_positiveReward():
    """[[72, 12, 6]]: l=6, m=6, A=x³+y+y², B=y³+x+x²"""
    import gymnasium as gym
    l, m = 6, 6
    env = gym.make('qecc/bbcode-v0', l=l, m=m,
                   evaluationDecoderFunction=_fast_decoder,
                   errorRange=TEST_ERROR_RANGE,
                   minimumNumberOfLogicalQubits=1)
    env.reset()
    _, reward, *_ = env.step(_make_action(l, m, [3], [1, 2], [1, 2], [3]))
    assert reward > 0


def test_IBM_90_8_10_positiveReward():
    """[[90, 8, 10]]: l=15, m=3, A=x⁹+y+y², B=1+x²+x⁷"""
    import gymnasium as gym
    l, m = 15, 3
    env = gym.make('qecc/bbcode-v0', l=l, m=m,
                   evaluationDecoderFunction=_fast_decoder,
                   errorRange=TEST_ERROR_RANGE,
                   minimumNumberOfLogicalQubits=1)
    env.reset()
    _, reward, *_ = env.step(_make_action(l, m, [9], [1, 2], [0, 2, 7], []))
    assert reward > 0


def test_IBM_108_8_10_positiveReward():
    """[[108, 8, 10]]: l=9, m=6, A=x³+y+y², B=y³+x+x²"""
    import gymnasium as gym
    l, m = 9, 6
    env = gym.make('qecc/bbcode-v0', l=l, m=m,
                   evaluationDecoderFunction=_fast_decoder,
                   errorRange=TEST_ERROR_RANGE,
                   minimumNumberOfLogicalQubits=1)
    env.reset()
    _, reward, *_ = env.step(_make_action(l, m, [3], [1, 2], [1, 2], [3]))
    assert reward > 0


def test_IBM_144_12_12_positiveReward():
    """[[144, 12, 12]]: l=12, m=6, A=x³+y+y², B=y³+x+x²"""
    import gymnasium as gym
    l, m = 12, 6
    env = gym.make('qecc/bbcode-v0', l=l, m=m,
                   evaluationDecoderFunction=_fast_decoder,
                   errorRange=TEST_ERROR_RANGE,
                   minimumNumberOfLogicalQubits=1)
    env.reset()
    _, reward, *_ = env.step(_make_action(l, m, [3], [1, 2], [1, 2], [3]))
    assert reward > 0


def test_IBM_288_12_18_positiveReward():
    """[[288, 12, 18]]: l=12, m=12, A=x³+y²+y⁷, B=y³+x+x²"""
    import gymnasium as gym
    l, m = 12, 12
    env = gym.make('qecc/bbcode-v0', l=l, m=m,
                   evaluationDecoderFunction=_fast_decoder,
                   errorRange=TEST_ERROR_RANGE,
                   minimumNumberOfLogicalQubits=1)
    env.reset()
    _, reward, *_ = env.step(_make_action(l, m, [3], [2, 7], [1, 2], [3]))
    assert reward > 0


if __name__ == "__main__":

    test_bbCodesEnvIsWorking()
    test_bbCodesEnvIsRegistered()
    test_envPassesBasicChecks()
    test_IBM_72_12_6()
    test_IBM_90_8_10()
    test_IBM_108_8_10()
    test_IBM_144_12_12()
    test_IBM_288_12_18()
