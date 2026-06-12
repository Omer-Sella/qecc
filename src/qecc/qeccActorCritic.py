import numpy as np
import torch.nn as nn
import torch
from torch.distributions import Categorical

MODELS_BOOLEAN_TYPE = np.bool

"""
qeccActorCritic
    Is a top module that uses:
    -- multi layer perceptron (explicitMLP)
    For the valuation, since the valuation is a simple function.
    It also uses
    -- qeccActor:
    for the policy. The policy is (usually, or in the future) a more complicated function, using potentially several:
        -- multi layer perceptron (explicitMLP)
"""


class explicitMLP(nn.Module):
    """
    explicitMLP creates a multi layer perceptron with explicit input and output lengths.
    if hiddenLayersLengths is not an empty list it will create hidden layers with the specified lengths as input lengths.
    default activation is the identity.
    """
    def __init__(self, firstLayerSize, lastLayerSize, hiddenLayersSpecification, intermediateActivation = nn.Identity, outputActivation = nn.Identity):
        super().__init__()
        lengths = [firstLayerSize] + hiddenLayersSpecification + [lastLayerSize]
        self.outputActivation = outputActivation
        layerList = []
        
        for l in range(len(lengths) - 1):
            if (l < (len(lengths) - 2)):
                activation = intermediateActivation
            else:
                activation = outputActivation
            layerList = layerList + [nn.Linear(lengths[l], lengths[l + 1]), activation()]
        self.layers = nn.ModuleList(layerList)
        self.outputDimension = lastLayerSize
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class qeccActor(nn.Module):
        def __init__(self, observationSpaceType, observationSpaceSize, actionSpaceType, actionSpaceSize, hiddenEncoderSize, maximumNumberOfHotBits, hiddenLayerParameters, actorCriticDevice = 'cpu'):
            super().__init__()
            self.observationSpaceType = observationSpaceType
            self.observationSpaceSize = observationSpaceSize
            self.actionSpaceType = actionSpaceType
            self.actionSpaceSize = actionSpaceSize
            self.device = actorCriticDevice
            self.maximumNumberOfHotBits = maximumNumberOfHotBits
            self.hiddenEncoderSize = hiddenEncoderSize
            self.rowCoordinateRange = 2
            self.columnCoordinateRange = 16
            self.circulantSize = 511
            self.defaultHiddenLayerSizes = [64]
            self.defaultActivation = nn.Identity
            self.encoder = explicitMLP(observationSpaceSize, hiddenEncoderSize, [hiddenEncoderSize, hiddenEncoderSize])
            self.rowCoordinateModel = explicitMLP(self.hiddenEncoderSize, self.rowCoordinateRange, self.defaultHiddenLayerSizes)
            self.columnCoordinateModel = explicitMLP(self.hiddenEncoderSize + 1, self.columnCoordinateRange, self.defaultHiddenLayerSizes)
            self.numberOfHotBitsModel = explicitMLP(self.hiddenEncoderSize + 2, self.maximumNumberOfHotBits, self.defaultHiddenLayerSizes)
            self.kHotVectorGenerator = explicitMLP(self.circulantSize, self.circulantSize, self.defaultHiddenLayerSizes)
            self.encoder2 = explicitMLP(self.hiddenEncoderSize + 3, self.circulantSize, self.defaultHiddenLayerSizes)
            self.to(actorCriticDevice)
        
        def actorActionToEnvAction(self, actorAction):
            i, j, k, hotCoordinates = actorAction
            """
            The actor is expected to produce i, j, and up to k coordinates which will be hot.
            The environment is expecting i,j and a binary vector.
            """
            binaryVector = np.zeros(self.circulantSize, dtype = MODELS_BOOLEAN_TYPE)
            binaryVector[hotCoordinates[0:k]] = 1
            environmentStyleAction = [i, j, binaryVector]
            return environmentStyleAction
    
        def step(self, observations, action = None):
        
            # The step function has 3 modes:
            #   1. Training - this is where we use the model and sample from the distributions parametrised by the model.
            #   2. Most probable action - this is where we use the deterministic model and instead of sampling take the most probable action.
            #   3. Both observations AND actions are provided, in which case we are evaluating log probabilities
        
            # Observations batchSize X observationSpaceSize of type observationSpaceType

            if action is not None:
                action = torch.as_tensor(action, device = self.device)
        
            encodedObservation = self.encoder(observations)
            logitsForIChooser = self.rowCoordinateModel(encodedObservation)
            iCategoricalDistribution = Categorical(logits = logitsForIChooser)
            iDistributionEntropy = iCategoricalDistribution.entropy().unsqueeze(-1)
            if action is not None:
                i = action[:, 0]
            elif self.training:
                i = iCategoricalDistribution.sample()
            else:
                i = torch.argmax(logitsForIChooser)
               
        
                
            # Omer Sella: now we need to append i to the observations
            ## Omer Sella: when acting you need to sample and concat. When evaluating, you need to break the action into internal components and set to them.
            ## Then log probabilities are evaluated at the end (regardless of whether this was sampled or given)
        
            i = i.float()
            iTensor = i.unsqueeze(-1)
            iAppendedObservations = torch.cat([encodedObservation, iTensor], dim = -1)
            logitsForJChooser = self.columnCoordinateModel(iAppendedObservations)
            jCategoricalDistribution = Categorical(logits = logitsForJChooser)
            jDistributionEntropy = jCategoricalDistribution.entropy().unsqueeze(-1)

            if action is not None:
                j = action[:, 1]
            elif self.training:    
                j = jCategoricalDistribution.sample()
            else:
                j = torch.argmax(logitsForJChooser)
                
            # Omer Sella: now we need to append j to the observations
            j = j.float()
            jTensor = j.unsqueeze(-1)
            jAppendedObservations = torch.cat([iAppendedObservations, jTensor], dim = -1)
            logitsForKChooser = self.numberOfHotBitsModel(jAppendedObservations)
            kCategoricalDistribution = Categorical(logits = logitsForKChooser)
            kDistributionEntropy = kCategoricalDistribution.entropy().unsqueeze(-1)
        
            if action is not None:
                k = action[:, 2]
            elif self.training:
                k = kCategoricalDistribution.sample()
                #Omer Sella: k can't be 0
                k = kCategoricalDistribution.sample() + 1
            else:
                #k = torch.argmax(logitsForKChooser)
                #Omer Sella: k can't be 0
                k = torch.argmax(logitsForKChooser) + 1
        
            k = k.float()
            kTensor = k.unsqueeze(-1)
            kAppendedObservations = torch.cat([jAppendedObservations, kTensor], dim = -1)
            setEncodedStuff = self.encoder2(kAppendedObservations)
        
        
            
            
        
            # In this part we choose k coordinates, where: k <= maximumNumberOfHotBits <= circulantSize 
            # In practice we choose maximumNumberOfHotBits coordinates, and use only the first k of them
            
            if action is not None:
                coordinates = action[:, 3 : 3 + self.maximumNumberOfHotBits]
                numberOfObservations = coordinates.shape[0]
                print(coordinates.shape[0])
                coordinateEntropies = torch.zeros((numberOfObservations, self.maximumNumberOfHotBits))
                logProbCoordinates = torch.zeros((numberOfObservations, self.maximumNumberOfHotBits))
                
                
                idx = 0
                while idx < self.maximumNumberOfHotBits:
                    logitsForCoordinateChooser = self.kHotVectorGenerator(setEncodedStuff)
                    circulantSizeCategoricalDistribution = Categorical(logits = logitsForCoordinateChooser)
                    newCoordinate = coordinates[:, idx]
                    logProbCoordinates[:, idx] = circulantSizeCategoricalDistribution.log_prob(newCoordinate)
                    coordinateEntropies[:, idx] = circulantSizeCategoricalDistribution.entropy()# Omer Sella: commented this: .unsqueeze(-1)
                    setEncodedStuff = setEncodedStuff + logitsForCoordinateChooser
                    idx = idx + 1
            elif self.training:
                coordinateEntropies = torch.zeros(self.maximumNumberOfHotBits)
                logProbCoordinates = torch.zeros(self.maximumNumberOfHotBits)
                coordinates = -1 * np.ones(self.maximumNumberOfHotBits)
                idx = 0
                while idx < self.maximumNumberOfHotBits:
                    logitsForCoordinateChooser = self.kHotVectorGenerator(setEncodedStuff)
                    circulantSizeCategoricalDistribution = Categorical(logits = logitsForCoordinateChooser)
                    newCoordinate = circulantSizeCategoricalDistribution.sample()
                    coordinates[idx] = newCoordinate
                    logProbCoordinates[idx] = circulantSizeCategoricalDistribution.log_prob(newCoordinate)
                    coordinateEntropies[idx] = circulantSizeCategoricalDistribution.entropy().unsqueeze(-1)
                    setEncodedStuff = setEncodedStuff + logitsForCoordinateChooser
                    idx = idx + 1
            else:
                coordinateEntropies = torch.zeros(self.maximumNumberOfHotBits)
                logProbCoordinates = torch.zeros(self.maximumNumberOfHotBits)
                coordinates = -1 * np.ones(self.maximumNumberOfHotBits)
                idx = 0
                while idx < self.maximumNumberOfHotBits:
                    logitsForCoordinateChooser = self.kHotVectorGenerator(setEncodedStuff)
                    circulantSizeCategoricalDistribution = Categorical(logits = logitsForCoordinateChooser)
                    newCoordinate = torch.argmax(logitsForCoordinateChooser)
                    coordinates[idx] = newCoordinate
                    logProbCoordinates[idx] = circulantSizeCategoricalDistribution.log_prob(newCoordinate)
                    coordinateEntropies[idx] = circulantSizeCategoricalDistribution.entropy().unsqueeze(-1)
                    setEncodedStuff = setEncodedStuff + logitsForCoordinateChooser
                    idx = idx + 1
                
                    
            #log probs
            logpI = iCategoricalDistribution.log_prob(i).unsqueeze(-1)
            logpJ = jCategoricalDistribution.log_prob(j).unsqueeze(-1)#.sum(axis = -1)
            #Omer Sella: remember that you added 1 to k so to get the log prob reduce 1
            logpK = kCategoricalDistribution.log_prob(k-1).unsqueeze(-1)#.sum(axis = -1)
            
            
            if action is None:
                i = np.int32(i.item())
                j = np.int32(j.item())
                k = np.int32(k.item())
                coordinates = np.int32(coordinates)
                
                
            return i, j, k, coordinates, logpI, logpJ, logpK, logProbCoordinates, iDistributionEntropy, jDistributionEntropy, kDistributionEntropy, coordinateEntropies


class qeccActorCritic(nn.Module):
    """
    A qecc actor critic wraps a policy (actor) and a valuation (critic) together.
    It provides a step function that accepts an observation, and returns an action and an expected value (reward)


    """
    def __init__(self, observationSpaceType, observationSpaceSize, actionSpaceType, actionSpaceSize, hiddenEncoderSize, maximumNumberOfHotBits, hiddenLayerParameters, actorCriticDevice = 'cpu'):
        super().__init__()
        # Initialize a policy 
        self.policy = qeccActor(observationSpaceType, observationSpaceSize, actionSpaceType, actionSpaceSize, hiddenEncoderSize, maximumNumberOfHotBits, hiddenLayerParameters, actorCriticDevice)
        # Initialize a valuation
        self.valuation  = explicitMLP(observationSpaceSize, observationSpaceSize, hiddenLayerParameters)

    def step(self, observations, actions = None):
        actions, logProbabilityAction =  self.policy.step(observations, actions)
        value = self.valuation(observations)
        return actions, value, logProbabilityAction
