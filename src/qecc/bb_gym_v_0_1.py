import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.integrate import trapezoid
from qecc.polynomialCodes import generateBicycleCode, generateABmatrices, bicycleCodeFromAB
from qecc.logicals import calculateCodeDimension
from qecc.gf4 import integerToDualBinary
from ldpc import BpOsdDecoder
from qecc.logicals import computeLogicals
import json
import os
from datetime import datetime, timezone


INT_DATA_TYPE = np.int16

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

    def __init__(self, l, m, errorRange = np.linspace(0.0001,0.1,10), minimumNumberOfLogicalQubits = 6, render_mode = None, numberOfSamples = 50, numberOfIterations = 50, rewardEngineering = None, seed = 0, codeLogging = True):
        
        self.codeLogging = codeLogging
        self.render_mode = render_mode # There is no rendering, but we have to accept and store it to comply with gymnasium spec.
        self.minimumNumberOfLogicalQubits = minimumNumberOfLogicalQubits
        self.numberOfIterations = numberOfIterations
        self.numberOfSamples = numberOfSamples
        self.ms_scaling_factor = 0.625
        self._l = l
        self._m = m
        self.seed = seed # Omer: WARNING ! The assumption is that either on init, or later in reset, or parallelEnv or collector will set a seed.
        self.errorRange = errorRange
        if any(a >= b for a, b in zip(errorRange, errorRange[1:])):
            raise ValueError(
                f"errorRange must be strictly increasing (e.g. [0.001, 0.01, 0.1]); got {list(errorRange)}"
            )
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
        
        
        self.Hx, self.Hz = bicycleCodeFromAB(self.A, self.B)
        
        # self.observation_space = spaces.Dict(
        #      {
        #          "Hx": spaces.MultiBinary([self._l*self._m, self._l*self._m *2]),
        #          "Hz": spaces.MultiBinary([self._l*self._m, self._l*self._m *2]),
        #      }
        #  )
        self.flatObservationSize = ((self._l * self._m) ** 2) * 2
        
        self.observation_space = spaces.MultiBinary(self.flatObservationSize)
        if rewardEngineering:
            def rewardEngineeringFunction(plainReward):
                return np.exp(np.exp(plainReward) - 1) -1 # Once we hit the number of necessary qubits, we exponent twice to encourage better performing codes.
            self.rewardEngineering = rewardEngineeringFunction
        else:
            self.rewardEngineering = lambda x: x
    
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
        if seed is None:
            self.seed = self.seed + 1
        else:
            self.seed = seed
        super().reset(seed = self.seed)
        self.aX = self.aX * 0
        self.aY = self.aY * 0
        self.bX = self.bX * 0
        self.bY = self.bY * 0
        self.A, self.B = generateABmatrices(self._l, self._m,
                                            np.where(self.aX !=0)[0], 
                                               np.where(self.aY !=0)[0], 
                                               np.where(self.bX !=0)[0], 
                                               np.where(self.bY !=0)[0])
        
        self.Hx, self.Hz = bicycleCodeFromAB(self.A, self.B)
        observation = self._getObservation()
        #info = self._getInfo()
        info = {}
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

        self.Hx, self.Hz = bicycleCodeFromAB(self.A, self.B)
        # Omer: check that the resulting code admits the necessary logical qubits
        
        self.numberOfLogicalQubits = calculateCodeDimension(self.Hx, self.Hz)
        if  self.numberOfLogicalQubits >= self.minimumNumberOfLogicalQubits:
            #seedForEvaluation = self.np_random.integers(0, 2**32 - 1) #Changed to environment seed
            self.seed = self.seed + 1
            logicalErrorRate = self.decoderEvaluation(self.seed)
            reward = self.rewardEngineering(self._calculateReward(logicalErrorRate))
        else:
            reward = (self.numberOfLogicalQubits - self.minimumNumberOfLogicalQubits) / self.minimumNumberOfLogicalQubits 

        terminated = False
        observation = self._getObservation()
        info = {}#None
        return observation, reward, terminated, False, info
    
    def decoderEvaluation(self, seed):

        
        localRandom = np.random.RandomState(seed) 
        logicalX, logicalZ = computeLogicals(self.Hx, self.Hz)

        bpDecoderHx=BpOsdDecoder(self.Hx,#the parity check matrix
                                        error_rate=1.0,
                                        #channel_probs= initialValues[:,1], #assign error_rate to each qubit. This will override "error_rate" input variable
                                        max_iter=self.numberOfIterations, #the maximum number of iterations for BP)
                                        bp_method="ms",
                                        ms_scaling_factor=self.ms_scaling_factor, #min sum scaling factor. If set to zero the variable scaling factor method is used
                                        osd_method="osd0", #the OSD method. Choose from:  1) "osd_e", "osd_cs", "osd0"
                                        osd_order=0 #the osd search depth
                                        )
        bpDecoderHz=BpOsdDecoder(self.Hz,#the parity check matrix
                                        error_rate=1.0,
                                        #channel_probs= initialValues[:,1], #assign error_rate to each qubit. This will override "error_rate" input variable
                                        max_iter=self.numberOfIterations, #the maximum number of iterations for BP)
                                        bp_method="ms",
                                        ms_scaling_factor=self.ms_scaling_factor, #min sum scaling factor. If set to zero the variable scaling factor method is used
                                        osd_method="osd0", #the OSD method. Choose from:  1) "osd_e", "osd_cs", "osd0"
                                        osd_order=0 #the osd search depth
                                        )
        logicalErrorRate = np.zeros(len(self.errorRange))
        decoderFailure = np.zeros(len(self.errorRange))
        for i in range(len(self.errorRange)):
            p = float(self.errorRange[i])
            bpDecoderHx.error_rate = p
            bpDecoderHz.error_rate = p
            logicalErrorRate[i] = 0
            for _ in range(self.numberOfSamples):
                #Sample (some number of times) an error, which is a vector over {0,1,2,3} representing I,X,Z,Y (to be consistent check with the documentation in gf4.py)
                error = localRandom.choice([0,1,2,3], size=self.Hx.shape[1], replace=True, p=[1 - 3*p, p, p, p])
                errorX, errorZ = integerToDualBinary(error)
                #Calculate the syndrome for this error
                estimatedErrorX = bpDecoderHz.decode(self.Hz.dot(errorX)%2)
                estimatedErrorZ = bpDecoderHx.decode(self.Hx.dot(errorZ)%2)
                residualErrorX = (estimatedErrorX + errorX) % 2
                residualErrorZ = (estimatedErrorZ + errorZ) % 2
                # Check whether the residual error gives 0 syndrome:
                if not ( np.all((np.dot(self.Hx,residualErrorZ)) % 2 ==0) and np.all((np.dot(self.Hz, residualErrorX)%2) == 0)):
                    #print("Decoder failure: the residual error does not give 0 syndrome, meaning the decoder is wrong")
                    decoderFailure[i] += 1
                else: # So we are in the case that the residual error commutes with all stabilizers, i.e., it is in the normalizer. So let's check if it is a stabilizer (commutes with all logicals), or a logical error (anticommutes with some logical operator)
                    if not ( np.all((np.dot(logicalZ,residualErrorX) % 2 )== 0) and np.all((np.dot(logicalX, residualErrorZ) % 2)==0)):
                        #print(f"Logical error: the residual error commutes with all stabilizers but anticommutes with some logical operator")
                        logicalErrorRate[i] += 1
              
        
        if self.codeLogging:
            logCodeEvaluation(self._l, self._m, self.aX, self.aY, self.bX, self.bY, self.numberOfLogicalQubits,
                      self.errorRange, logicalErrorRate, decoderFailure,
                      self.numberOfSamples, self.seed, "dual binary bposd 0", "depolarizing",
                      runId=None, logDirectory=None)

        return (logicalErrorRate + decoderFailure)/self.numberOfSamples
    
    def _calculateReward(self, logicalErrorRate):      
        reward = trapezoid(1 - logicalErrorRate , self.errorRange)
        return reward
    
    def getSeed(self):
        return self.seed




def makeTestAction_6_6():
    l = 6
    m = 6
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
    action = np.concatenate([aX,aY,bX,bY])
    return action
    


def logCodeEvaluation(l, m, aX, aY, bX, bY, numberOfLogicalQubits,
                      errorRange, logicalErrorCounts, decoderFailureCounts,
                      numberOfSamples, seed, decoderConfig, noiseModel,
                      runId=None, logDirectory=None):
    """Append one decoder-evaluation record to a per-process JSON-lines file."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runId": runId,
        "l": int(l),
        "m": int(m),
        "aX": [int(i) for i in aX],
        "aY": [int(i) for i in aY],
        "bX": [int(i) for i in bX],
        "bY": [int(i) for i in bY],
        "numberOfLogicalQubits": int(numberOfLogicalQubits),
        "errorRange": [float(p) for p in errorRange],
        "logicalErrorCounts": [int(c) for c in logicalErrorCounts],
        "decoderFailureCounts": [int(c) for c in decoderFailureCounts],
        "numberOfSamples": int(numberOfSamples),
        "seed": int(seed),
        "decoder": decoderConfig,
        "noiseModel": noiseModel,
    }
    if logDirectory is None:
        logDirectory = os.environ.get("QECC_DATA", ".")
    # One file per process: safe under ParallelEnv multiprocessing on Windows.
    path = os.path.join(logDirectory, f"codeEvaluations_{os.getpid()}.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":

    # Some basic examples of how to use:
    l = 6
    m = 6
    device = 'cpu'
    
    # Check the environment works with GymEnv
    from torchrl.envs.libs.gym import GymEnv
    from torchrl.envs.utils import check_env_specs
    base_env = GymEnv("qecc/bbcode-ldpc-v0", l = 6, m = 6, errorRange = np.linspace(0.0001,0.1,10), minimumNumberOfLogicalQubits = 6)

    check_env_specs(base_env)
    env = gym.make('qecc/bbcode-ldpc-v0', l = 6, m = 6, errorRange = np.linspace(0.0001,0.1,10), minimumNumberOfLogicalQubits = 6)

    
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

