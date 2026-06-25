import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.integrate import trapezoid
from qecc.polynomialCodes import generateBicycleCode, generateABmatrices
from qecc.utils import decoderEvaluator
from qecc.logicals import calculateCodeDimension
import copy
INT_DATA_TYPE = np.int16
NEGATIVE_REWARD = -1
# Following https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/

class bicycleBivariateCodeEnvironment(gym.Env):
    metadata = {"render_modes" : []}
    """
    A gymnasium environment to learn bicycle bivariate codes as described in Bivariate Bicycle codes from High-threshold and low-overhead fault-tolerant quantum memory
    Some parameters for BB codes from https://arxiv.org/pdf/2308.07915
    [[72, 12, 6]]
    [[90, 8, 10]]
    [[108, 8, 10]]
    [[144, 12,12]]
    [[288, 12,18]]
    """

    def __init__(self, l, m, evaluationDecoderFunction, errorRange = [0.1, 0.06, 0.01, 0.006, 0.001, 0.0006, 0.0001 ], minimumNumberOfLogicalQubits = 6, render_mode = None):
        
        self.render_mode = render_mode # There is no rendering, but we have to accept and store it to comply with gymnasium spec.
        self.minimumNumberOfLogicalQubits = minimumNumberOfLogicalQubits
        self.decoder = evaluationDecoderFunction
        self._l = l
        self._m = m
        self.seed = None
        self.errorRange = errorRange
        # The action space is a flat array containing [aX,bX,aY,bY] in that order.
        self.action_space = spaces.MultiBinary(4 * l * m)

        
        self.aX = np.zeros(l*m, INT_DATA_TYPE)
        self.bX = np.zeros(l*m, INT_DATA_TYPE)
        self.aY = np.zeros(l*m, INT_DATA_TYPE)
        self.bY = np.zeros(l*m, INT_DATA_TYPE)
        
        self.A, self.B = generateABmatrices(self._l, self._m,
                                            np.where(self.aX !=0)[0], 
                                            np.where(self.aY !=0)[0], 
                                            np.where(self.bX !=0)[0], 
                                            np.where(self.bY !=0)[0])
        
        
        self.Hx, self.Hz = generateBicycleCode(self._l, self._m, 
                                               np.where(self.aX !=0)[0], 
                                               np.where(self.aY !=0)[0], 
                                               np.where(self.bX !=0)[0], 
                                               np.where(self.bY !=0)[0])
        
        # self.observation_space = spaces.Dict(
        #      {
        #          "Hx": spaces.MultiBinary([self._l*self._m, self._l*self._m *2]),
        #          "Hz": spaces.MultiBinary([self._l*self._m, self._l*self._m *2]),
        #      }
        #  )
        self.flatObservationSize = ((self._l * self._m) ** 2) * 2
        
        self.observation_space = spaces.MultiBinary(self.flatObservationSize)
    
    # Gymnasium spec requires a render function and a close function:
    def render(self):
        pass
    def close(self):
        pass

    def _getObservation(self):
        # Omer: There is a warning about the obs returned not within observation space. Tried int8 but didn't work.
        #return {"Hx": np.int8(self.Hx), "Hz": np.int8(self.Hz)}
        #return {"Hx": self.Hx, "Hz": self.Hz}
        #return np.vstack((self.Hx, self.Hz)).flatten() 
        return np.vstack((self.A, self.B)).flatten().astype(np.int8)
    
    def reset(self, seed=None, options = None):
        super().reset(seed = seed)
        self.seed = seed
        self.aX = self.aX * 0
        self.aY = self.aY * 0
        self.bX = self.bX * 0
        self.bY = self.bY * 0
        self.A, self.B = generateABmatrices(self._l, self._m,
                                            np.where(self.aX !=0)[0], 
                                               np.where(self.aY !=0)[0], 
                                               np.where(self.bX !=0)[0], 
                                               np.where(self.bY !=0)[0])
        
        self.Hx, self.Hz = generateBicycleCode(self._l,self._m, 
                                               np.where(self.aX !=0)[0], 
                                               np.where(self.aY !=0)[0], 
                                               np.where(self.bX !=0)[0], 
                                               np.where(self.bY !=0)[0])

        observation = self._getObservation()
        #info = self._getInfo()
        info = {}#None
        return observation, info

    
    def step(self, action):
#        super().step(action)
        # Unpack action from flat action
        self.aX = action[0 : (self._l * self._m) ]
        self.aY = action[(self._l * self._m) : 2 * (self._l * self._m)]
        self.bX = action[2*(self._l * self._m) : 3 * (self._l * self._m)]
        self.bY = action[3*(self._l * self._m) : 4 * (self._l * self._m)]
        
        self.A, self.B = generateABmatrices(self._l, self._m, 
                                            np.where(self.aX !=0)[0], 
                                            np.where(self.aY !=0)[0], 
                                            np.where(self.bX !=0)[0], 
                                            np.where(self.bY !=0)[0])

        self.Hx, self.Hz = generateBicycleCode(self._l,self._m, 
                                               np.where(self.aX !=0)[0], 
                                               np.where(self.aY !=0)[0], 
                                               np.where(self.bX !=0)[0], 
                                               np.where(self.bY !=0)[0])
        #self.Hx = Hx.astype(int)
        #self.Hz = Hz.astype(int)
        # Omer: check that the resulting code admits the necessary logical qubits
        if calculateCodeDimension(self.Hx, self.Hz) > self.minimumNumberOfLogicalQubits:    
            logicalErrorRate, decoderFailureRate = self.decoder(self.Hx, self.Hz, self.errorRange, seed = self.seed)
            reward = self._calculateReward(logicalErrorRate, decoderFailureRate)
        else:
            reward = NEGATIVE_REWARD

        terminated = False
        observation = self._getObservation()
        info = {}#None
        return observation, reward, terminated, False, info
    

    
    def _calculateReward(self, logicalErrorRate, decoderFailureRate, numberOfIterations = 10):      
        #ber = (logicalErrorRate + decoderFailureRate)
        outputBER = logicalErrorRate + decoderFailureRate
        # snr = np.array(copy.copy(self.errorRange))
        # itr = 0 #Omer Sella: place holder - in the future we may want to return the iteration at which earlyStopping happened
        # while itr < numberOfIterations:
        #     #p = np.polyfit(self.errorRange, ber, 1)
        #     p = np.polyfit(snr, ber, 1)
        #     trendP = np.poly1d(p)
        #     mask = np.where(trendP(snr) > 0)[0]
        #     ber = ber[mask]
        #     snr = snr[mask]
        #     itr = itr + 1
        #     # Omer Sella: 16/06/2021 decided to use np polynomials. Also changed the reward to the area between
        #     # the constant 1 and the fitted line.            
        # pConst = np.poly1d([1])
        # p1 = np.poly1d(p)
        # pTotalInteg = (pConst - p1).integ()
        # reward = pTotalInteg(self.errorRange[-1]) - pTotalInteg(self.errorRange[0])
        reward = trapezoid(1 - outputBER, self.errorRange)
        return reward

def exampleDecoderFunction(Hx,Hz,errorRange, seed = None):
    from qecc.minSum import ldpcDecoderWrapper
    from qecc.utils import decoderEvaluator

    numberOfSamples = 30
    logicalErrors, decoderFailures =  decoderEvaluator(decoderFunction = ldpcDecoderWrapper, dualBinary = True, Hx = Hx, Hz = Hz, errorRange = errorRange, decoderStoppingCriterion = 50, numberOfSamples = numberOfSamples, seed = seed)
    #return {key: value/numberOfSamples for key,value in logicalErrors.items()} , {key: value/numberOfSamples for key,value in decoderFailures.items()}
    return logicalErrors/numberOfSamples, decoderFailures/numberOfSamples


def exampleDecoderFunction2(Hx,Hz,errorRange, seed = None):
    from qecc.utils import decoderEvaluator, wrapperForRoffesLdpc, binaryDecoderToDualBinaryDecoderWrapper
    decoder = binaryDecoderToDualBinaryDecoderWrapper(wrapperForRoffesLdpc) # So this will be ms_scaling_factor = 0.625, min-sum, osd0
    NUMBER_OF_SAMPLES = 100
    NUMBER_OF_DECODER_ITERATIONS = 50
    logicalErrors, decoderFailures =  decoderEvaluator(decoderFunction = decoder, dualBinary = True, Hx = Hx, Hz = Hz, errorRange = errorRange, decoderStoppingCriterion = NUMBER_OF_DECODER_ITERATIONS, numberOfSamples = NUMBER_OF_SAMPLES, seed = seed)
    return logicalErrors / NUMBER_OF_SAMPLES, decoderFailures / NUMBER_OF_SAMPLES
if __name__ == "__main__":
    l = 6
    m = 6
    device = 'cpu'
    
    # Check the environment works with GymEnv
    from torchrl.envs.libs.gym import GymEnv
    from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
    base_env = GymEnv("qecc/bbcode-v0", device=device, l = 6, m = 6, evaluationDecoderFunction = exampleDecoderFunction, errorRange = [0.1, 0.06, 0.01, 0.006, 0.001, 0.0006, 0.0001 ], minimumNumberOfLogicalQubits = 6)

    check_env_specs(base_env)
    env = gym.make('qecc/bbcode-v0', l = 6, m = 6, evaluationDecoderFunction = exampleDecoderFunction, errorRange = [0.1, 0.06, 0.01, 0.006, 0.001, 0.0006, 0.0001 ])

    #env = gym.make('qecc/bbcode-v0', l = 6, m = 6, evaluationDecoderFunction = dualRoffeDecoder, errorRange = [0.1, 0.06, 0.01, 0.006, 0.001, 0.0006, 0.0001 ])
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
    print(action)
    observation = env.step(action = action)

