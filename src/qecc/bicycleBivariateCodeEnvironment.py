import numpy as np
import gymnasium as gym
from gymnasium import spaces
from qecc.polynomialCodes import generateBicycleCode
from qecc.utils import decoderEvaluator
INT_DATA_TYPE = np.int16
class bicycleBivariateCodeEnvironment(gym.Env):
    """
    A gymnasium environment to learn bicycle bivariate codes as described in Bivariate Bicycle codes from High-threshold and low-overhead fault-tolerant quantum memory
    """

    def __init__(self, l, m, evaluationDecoder, errorRange):
        self.decoder = evaluationDecoder
        self.errorRange = errorRange
        self._l = l
        self._m = m
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
    
    def reset(self):
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
        info = self._getInfo()
        return observation, info

    
    def step(self, action):
        logicalErrorRate, decoderFailureRate = evaluationDecoder(self.Hx, self.Hz, errorRange)
        reward = self._calculateReward(logicalErrorRate, decoderFailureRate)
        terminated = False
        observation = self._getObservation()
        info = None
        return observation, reward, terminated, False, info
    
    def _getObservation(self):
        return {"Hx": self.Hx, "Hz": self.Hz}