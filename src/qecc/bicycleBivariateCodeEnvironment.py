import numpy as np
import gymnasium as gym
from gymnasium import spaces
from qecc.polynomialCodes import generateBicycleCode
from qecc.utils import decoderEvaluator
import copy
INT_DATA_TYPE = np.int16
class bicycleBivariateCodeEnvironment(gym.Env):
    """
    A gymnasium environment to learn bicycle bivariate codes as described in Bivariate Bicycle codes from High-threshold and low-overhead fault-tolerant quantum memory
    """

    def __init__(self, l, m, evaluationDecoderFunction, errorRange):
        self.decoder = evaluationDecoderFunction
        self._l = l
        self._m = m
        self.seed = None
        self.errorRange = errorRange
        self.action_space = spaces.Tuple( (spaces.MultiBinary(l*m), 
                                           spaces.MultiBinary(l*m), 
                                           spaces.MultiBinary(l*m), 
                                           spaces.MultiBinary(l*m)))
        
        self.aX = np.zeros(l*m, INT_DATA_TYPE)
        self.bX = np.zeros(l*m, INT_DATA_TYPE)
        self.aY = np.zeros(l*m, INT_DATA_TYPE)
        self.bY = np.zeros(l*m, INT_DATA_TYPE)
        
        self.Hx, self.Hz = generateBicycleCode(self._l, self._m, 
                                               np.where(self.aX !=0)[0], 
                                               np.where(self.aY !=0)[0], 
                                               np.where(self.bX !=0)[0], 
                                               np.where(self.bY !=0)[0])
        
        self.observation_space = spaces.Dict(
            {
                "Hx": spaces.MultiBinary([self._l*self._m, self._l*self._m,]),
                "Hz": spaces.MultiBinary([self._l*self._m, self._l*self._m,]),
            }
        )

        
    
    def reset(self, seed = None, options = None):
        super().reset(seed = seed)
        self.aX = self.aX * 0
        self.aY = self.aY * 0
        self.bX = self.bX * 0
        self.bY = self.bY * 0
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
        
        self.aX = action[0]
        self.aY = action[1]
        self.bX = action[2]
        self.bY = action[3]
        
        self.Hx, self.Hz = generateBicycleCode(self._l,self._m, 
                                               np.where(self.aX !=0)[0], 
                                               np.where(self.aY !=0)[0], 
                                               np.where(self.bX !=0)[0], 
                                               np.where(self.bY !=0)[0])
        #self.Hx = Hx.astype(int)
        #self.Hz = Hz.astype(int)
        
        logicalErrorRate, decoderFailureRate = self.decoder(self.Hx, self.Hz, self.errorRange)
        reward = self._calculateReward(logicalErrorRate, decoderFailureRate)
        terminated = False
        observation = self._getObservation()
        info = {}#None
        return observation, reward, terminated, False, info
    
    def _getObservation(self):
        # Omer: There is a warning about the obs returned not within observation space. Tried int8 but didn't work.
        #return {"Hx": np.int8(self.Hx), "Hz": np.int8(self.Hz)}
        return {"Hx": self.Hx, "Hz": self.Hz}
    
    def _calculateReward(self, logicalErrorRate, decoderFailureRate, numberOfIterations = 10):      
        #ber = (logicalErrorRate + decoderFailureRate)
        ber = np.array([logicalErrorRate[k] + decoderFailureRate[k] for k in logicalErrorRate.keys()])    
        snr = np.array(copy.copy(self.errorRange))
        itr = 0 #Omer Sella: place holder - in the future we may want to return the iteration at which earlyStopping happened
        while itr < numberOfIterations:
            p = np.polyfit(self.errorRange, ber, 1)
            trendP = np.poly1d(p)
            mask = np.where(trendP(snr) > 0)[0]
            ber = ber[mask]
            snr = snr[mask]
            itr = itr + 1
            # Omer Sella: 16/06/2021 decided to use np polynomials. Also changed the reward to the area between
            # the constant 1 and the fitted line.            
        pConst = np.poly1d([1])#p1 = np.poly1d(p)
        pTotalInteg = (pConst - p).integ()
        reward = pTotalInteg(self.errorRange[-1]) - pTotalInteg(self.errorRange[0])
        return reward
    
if __name__ == "__main__":
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
