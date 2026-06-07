def test_bbCodesEnvIsWorking():
    
    from qecc.minSum import ldpcDecoderWrapper
    from qecc.utils import decoderEvaluator
    import gymnasium as gym
    import numpy as np
    def decoderFunction(Hx,Hz,errorRange):
        numberOfSamples = 30
        logicalErrors, decoderFailures =  decoderEvaluator(decoderFunction = ldpcDecoderWrapper, dualBinary = True, Hx = Hx, Hz = Hz, errorRange = errorRange, decoderStoppingCriterion = 50, numberOfSamples = numberOfSamples)
        #return {key: value/numberOfSamples for key,value in logicalErrors.items()} , {key: value/numberOfSamples for key,value in decoderFailures.items()}
        return logicalErrors/numberOfSamples, decoderFailures/numberOfSamples
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
    action = np.hstack((np.hstack((aX,aY)), np.hstack((bX,bY))))
    observation = env.step(action = action)
    print(env.unwrapped.flatObservationSize)
    #print(dir(env))
    #print(observation)
    

def test_bbCodesEnvIsRegistered():
    import gymnasium as gym
    # Once qecc is imported, it's init file should have registered the environment. We can check that by checking the registry of gymnasium.
    import qecc
    allEnvs = gym.envs.registry.keys()
    assert "qecc/bbcode-v0" in allEnvs

def test_logicalXOnAction():
    """
    Test related to the following bug:
    if not ( np.all((np.dot(logicalZ,residualErrorX) % 2 )== 0) and np.all((np.dot(logicalX, residualErrorZ) % 2)==0)):
                                                                            ~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^
    ValueError: shapes (0,) and (72,) not aligned: 0 (dim 0) != 72 (dim 0)
    """
    import gymnasium as gym
    import qecc
    from qecc.bb_gym import exampleDecoderFunction
    import torch
    
    
    def environmentFunction():
        return gym.make('qecc/bbcode-v0', l = 6, m = 6, evaluationDecoderFunction = exampleDecoderFunction, errorRange = [0.01, 0.001])
    
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

if __name__ == "__main__":
    
    test_bbCodesEnvIsWorking()
    test_bbCodesEnvIsRegistered()
    test_logicalXOnAction()