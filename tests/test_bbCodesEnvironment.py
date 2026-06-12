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
    base_env = GymEnv("qecc/bbcode-v0", device=device, l = 6, m = 6, evaluationDecoderFunction = exampleDecoderFunction, errorRange = [0.01, 0.001], minimumNumberOfLogicalQubits = 6)
    check_env_specs(base_env)
    
    
    # Check with gym.make
    env = gym.make('qecc/bbcode-v0', l = 6, m = 6, evaluationDecoderFunction = exampleDecoderFunction, errorRange = [0.01, 0.001])
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
    pass



if __name__ == "__main__":
    
    test_bbCodesEnvIsWorking()
    test_bbCodesEnvIsRegistered()
    test_envPassesBasicChecks()