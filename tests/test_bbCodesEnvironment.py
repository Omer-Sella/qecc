def test_bbCodesEnvIsWorking():
    from qecc.minSum import ldpcDecoderWrapper
    from qecc.utils import decoderEvaluator
    def decoderFunction(Hx,Hz,errorRange):
        numberOfSamples = 30
        logicalErrors, decoderFailures =  decoderEvaluator(decoderFunction = ldpcDecoderWrapper, dualBinary = True, Hx = Hx, Hz = Hz, errorRange = errorRange, decoderStoppingCriterion = 50, numberOfSamples = numberOfSamples)
        return {key: value/numberOfSamples for key,value in logicalErrors.items()} , {key: value/numberOfSamples for key,value in decoderFailures.items()}
    l = 6
    m = 6
    env = gym.make('qecc/bbcode-v0', l = l, m = m, evaluationDecoderFunction = decoderFunction, errorRange = [0.01, 0.001])
    env.reset()
    aX = np.zeros(l*m)
    aY = np.zeros(l*m)
    bX = np.zeros(l*m)
    bY = np.zeros(l*m)
    aX[3] = 1
    aY[1]=1
    aY[2] = 1
    bX[1] = 1
    bX[2] = 1
    bY[3] = 1
    action = (aX,aY,bX,bY)
    observation = env.step(action = action)
    print(observation)

def test_bbCodesEnvIsRegistered():
    import qecc
    # Once qecc is imported, it's init file should have registered the environment. We can check that by checking the registry of gymnasium.
    import gymnasium as gym
    allEnvs = gym.envs.registry.keys()
    assert "bbcode-v0" in allEnvs