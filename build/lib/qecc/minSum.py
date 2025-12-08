## oss 08/07/2019
import numpy as np
import time
import concurrent.futures
import os
import copy
from numba import jit, int32, float32, types, typed, boolean, float64, int64
from numba.experimental import jitclass
#import math
from polynomialCodes import A1_HX
LDPC_LOCAL_PRNG = np.random.RandomState(7134066)
# LDPC_**_DATA_TYPE stores the data type over which all arithmetic is done.
# It is a nice way of changing the data type of the entire implementation at one place.
LDPC_DATA_TYPE = np.int64
LDPC_INT_DATA_TYPE = np.int64
LDPC_DECIMAL_DATA_TYPE = np.float64
LDPC_SEED_DATA_TYPE = np.int64
# Omer Sella: Major breakdown warning: the bool data type is used to create a mask. Replacing it with int32 breaks the decoder.
LDPC_BOOL_DATA_TYPE = boolean
# Omer Sella: seeds can be integers between 0 and 2**31 - 1
LDPC_MAX_SEED = 2**31 - 1
NUMBA_INT = int64
NUMBA_FLOAT = float64
NUMBA_BOOL = boolean



specForVariableNode = [
    ('identity', NUMBA_INT),
    ('fromChannel', NUMBA_FLOAT),
    ('presentState', NUMBA_FLOAT),
    
]
@jitclass(specForVariableNode)
class variableNode:
# a variableNode is merely a memory element. 
    def __init__(self, identity):
        self.identity = identity
       
        self.fromChannel = 0
        self.presentState = 0
        return
        
    def update(self, value):
        #assert(np.dtype(value) == LDPC_DECIMAL_DATA_TYPE)
        self.presentState = self.presentState + value
        return
    
    @property
    def id(self):
        return self.identity
    

## 
# key and value types for variableIDToIndex: 
key_value_types = (NUMBA_INT, NUMBA_INT)
specForCheckNode = [
    ('identity', NUMBA_FLOAT),
    ('numberOfConnectedVariableNodes', NUMBA_INT),
    ('incomingValues', NUMBA_FLOAT[:,:]),
    ('outgoingValues', NUMBA_FLOAT[:,:]),
    ('signVector', NUMBA_FLOAT[:]),
    ('magnitudeVector', NUMBA_FLOAT[:]),
    ('connectedVariableNodes', NUMBA_INT[:]),
    ('variableIDtoIndexDictionary', types.DictType(*key_value_types)),
    ('mask', LDPC_BOOL_DATA_TYPE[:]),
    ]
@jitclass(specForCheckNode)
class checkNode:
# A check node is where most of the logic is done.
# Every check node has an id (this is redundant in serial execution, but may make things easier when moving to an asynchronous implementation)
# A checkNode stores the values of the variableNodes in the incoming values with the sender identities.
# When all incoming values are received, it may calculate the sign vector of these values as well as the magnitude.
# Each outgoing message is a two entry array (a line in outGoingMessages), that contains a recipient id,
# and a value, specially calculated for each recipient, by taking the minimum of the incoming values 
# over all non-recipient ids, and multiplying by the product of signs (again, all signs other than the recipient sign)
    def __init__(self, identity, connectedVariableNodes):
        self.identity = identity
        numberOfConnectedVariableNodes = len(connectedVariableNodes)
        self.numberOfConnectedVariableNodes = numberOfConnectedVariableNodes
        self.incomingValues = np.zeros((numberOfConnectedVariableNodes,2), dtype = LDPC_DECIMAL_DATA_TYPE)
        self.incomingValues[:,0] = connectedVariableNodes
        
        self.outgoingValues = np.zeros((numberOfConnectedVariableNodes,2), dtype = LDPC_DECIMAL_DATA_TYPE)
        self.outgoingValues[:,0] = connectedVariableNodes

        self.signVector = np.ones(numberOfConnectedVariableNodes, dtype = LDPC_DECIMAL_DATA_TYPE)
        self.magnitudeVector = np.zeros(numberOfConnectedVariableNodes, dtype = LDPC_DECIMAL_DATA_TYPE)

        # Omer Sella: attempting to reduce decoder latency by removing the need of the function np.where
        self.variableIDtoIndexDictionary = typed.Dict.empty(*key_value_types)
        index = 0
        for cv in connectedVariableNodes:
            self.variableIDtoIndexDictionary[cv] = index
            index = index + 1
        self.mask = np.ones(self.numberOfConnectedVariableNodes, dtype = LDPC_BOOL_DATA_TYPE)
        
    def setSign(self):
        #Reset the sign vector from possible previous use
        self.signVector = self.signVector * 0 + 1
        #Set the sign of negatives to -1, positives and zero are left as 1
        self.signVector[ np.where(self.incomingValues[:,1] < 0) ] = -1
        sign = np.prod(self.signVector)
        return sign
        
    
    # Omer Sella: this function is replacing the receive function by using the dict data sturcture and removing np.where
    def receiveDictionary(self, variableID, value):
        index = self.variableIDtoIndexDictionary[variableID]
        self.incomingValues[index,1] = value
        return
    
    def receive(self, variableID, value):
        #print(self.incomingValues[:,0] == variableID)
        # Omer Sella: This is ultra-stupid, but it's a bug worth noting:
        # indexIn was created using np.where. Now, even though incomingValues[indexIn,0]
        # is a correct addressing and usage of indices (like MATLAB), the results of np.where is ([[someNumber]])
        # which is not an integer, and numba doesn't like this.
        #self.incomingValues[np.where(self.incomingValues[:,0] == variableID),1] = value - self.outgoingValues[np.where(self.outgoingValues[:,0] == variableID),1]
        indexIn = np.where(self.incomingValues[:,0] == variableID)[0][0]
        # Omer Sella: indexIn and indexOut are the same.
        #indexOut = np.where(self.outgoingValues[:,0] == variableID)[0][0]
        #assert(indexIn == indexOut)
        newValue = value - self.outgoingValues[indexIn,1]
        self.incomingValues[indexIn, 1] = newValue
        return
    
    def getValueForNodeDictionary(self, variableID):
        return self.outgoingValues[self.variableIDtoIndexDictionary[variableID],1] 
    
    
    def getValueForNode(self, variableID):
        value = self.outgoingValues[np.where(self.outgoingValues[:,0] == variableID)]
        value = value[0,1]
        return value

    def calcOutgoingValues(self):
        # Set the vector of signs (remember that the sign of 0 is 1), and obtain its product
        sign = self.setSign()
        # Set the vector of magnitudes using the numpy (standard) supplied abs function.
        self.magnitudeVector = np.abs(self.incomingValues[:,1])
        
        # Now we get cheeky: we locate the locations of the two lowest values in incoming values, say mindex_0 and m1.
        # They might be the same - we don't care.
        # Then we use just them for the outputs
        [m0,m1] = np.argsort(self.magnitudeVector)[0:2]
        smallest = self.magnitudeVector[m0]
        secondSmallest = self.magnitudeVector[m1]
        
        # Initialize outgoing values. Remember that sign is one of two options:  {1,-1}
        #self.outgoingValues[:,1] = self.outgoingValues[:,1] * 0 + sign
        #mask = np.ones(self.numberOfConnectedVariableNodes, dtype = LDPC_BOOL_DATA_TYPE)
        self.outgoingValues[:,1] = smallest * sign * self.signVector
        self.outgoingValues[m0,1] = secondSmallest * sign * self.signVector[m0]
        
        #for i in range(self.numberOfConnectedVariableNodes):
            # Set the mask to ignore the value at location i.
            #mask[i] = False
            # The following line should be read as if we are dividing rather than multiplying, but since we are dividing by either 1 or -1 it is the same as multiplyin. 
         #   self.outgoingValues[i,1] = self.outgoingValues[i,1] * self.signVector[i]
            # Once the sign was determined, we need to multiply by the minimum value, taken over all ABSOLUTE values EXCEPT at the i'th coordinate.
          #  self.outgoingValues[i,1] = self.outgoingValues[i,1] * np.min(self.magnitudeVector[mask])
            # Reset the mask to be the all True mask.
            #mask[i] = True
        return

        
specForLdpcDecoder = [
    ('H', NUMBA_FLOAT[:,:]),
    ('syndromeDecoding', NUMBA_BOOL),
    ('parityMatrix', NUMBA_FLOAT[:,:]),
    ('numberOfVariableNodes', NUMBA_INT),
    ('numberOfCheckNodes', NUMBA_INT),
    ('codewordLength', NUMBA_INT),
    ('softVector', NUMBA_FLOAT[:]),
    ('outgoingValues', NUMBA_FLOAT[:,:]),
    ('signVector', NUMBA_FLOAT[:]),
    ('magnitudeVector', NUMBA_FLOAT[:]),
    ('connectedVariableNodes', NUMBA_FLOAT[:]),
    ('variableNodes', types.ListType(variableNode)),
    ('checkNodes', types.ListType(checkNode)),
    ('checkNodeAddressBook', types.ListType(NUMBA_FLOAT)),
    ('variableNodeAddressBook', types.ListType(NUMBA_INT[:])),
    ('softVector', float64[:]),
    ]
#@jitclass(specForLdpcDecoder)
class ldpcDecoder:

    def __init__(self, H, syndromeDecoding = False):
        self.parityMatrix = H
        self.syndromeDecoding = syndromeDecoding
        m,n = H.shape
        self.numberOfVariableNodes = n
        self.numberOfCheckNodes = m
        self.codewordLength = n
        self.variableNodes = []
        self.checkNodes = []
        self.checkNodeAddressBook = []
        self.variableNodeAddressBook = []
        # Omer Sella: softVector is a place holder for the current state of a decoded vector (starting with information from the channel, and updating while iterating)
        self.softVector = np.zeros(self.numberOfVariableNodes, dtype = LDPC_DECIMAL_DATA_TYPE)
        for i in range(self.numberOfVariableNodes):
            addresses = np.where(self.parityMatrix[:,i] != 0)[0]
            self.variableNodeAddressBook.append(addresses)
            vn = variableNode(i)
            self.variableNodes.append(vn)
        for i in range(self.numberOfCheckNodes):
            # Omer Sella: Below there is a quick fix in the form of xyz[0]. It seems that the returned value is a sequence of length 1, where the first (and only) element is an array.
            addresses = np.where(self.parityMatrix[i,:] != 0)[0]
            self.checkNodeAddressBook.append(addresses)
            cn = checkNode(i, addresses)#self.checkNodeAddressBook[i])
            self.checkNodes.append(cn)
    
    def isCodeword(self, modulatedVector):
        binaryVector = np.where(modulatedVector == -1, 0, 1)
        #Omer Sella: H.dot(binaryVector) is the same as summation over axis 0 of H[:,binaryVector] so we convert float multiplication into indexing and summation
        # Omer Sella: The following are equivalent (use of where, use of asarray(condition))
        #print(np.where(binaryVector != 0))
        #print(np.asarray(binaryVector != 0).nonzero())
        #result1 = self.parityMatrix.dot(binaryVector) % 2
        
        result2 = self.parityMatrix[:,np.asarray(binaryVector != 0).nonzero()[0]]
        #print(result2.shape)
        #np.bitwise_xor.reduce(result)
        result2 = np.sum(result2, axis = 1) % 2
        
        #assert np.all(result1 == result2)
        
        if all(result2 == 0):
            status = 'Codeword'
        else:
            status = 'Not a codeword'
        return status, binaryVector

    def decoderSet(self, fromChannel):
        if self.syndromeDecoding:
            for i in range(self.numberOfCheckNodes):
                self.checkNodes[i].fromChannel = fromChannel[i]
                self.checkNodes[i].presentState = fromChannel[i]
        else:
            for i in range(self.numberOfVariableNodes):
                self.variableNodes[i].fromChannel = fromChannel[i]
                self.variableNodes[i].presentState = fromChannel[i]
        return

    
    #def vn2cnTemp(self, j, i, value):
    #    self.checkNodes[j].receive(self.variableNodes[i].identity, value)
    #    #self.checkNodes[j].receiveDictionary(self.variableNodes[i].identity, value)
    #    #assert ( idx1 == idx2 )
    #    return

    def variableNodesToCheckNodes(self):
        for i in range(self.numberOfVariableNodes):
            value = self.variableNodes[i].presentState
            # Send the current value to all check nodes connected to this variable node.
            recipientCheckNodes = self.variableNodeAddressBook[i]
            for j in recipientCheckNodes:
                
                # Omer Sella: Two things I need to change: 1. isn't the identity simply i ? 2. Isn't there a way to send the index of the 
                    self.checkNodes[j].receive(self.variableNodes[i].identity, value)
            # Once the current value was sent, reset it to 0
            self.variableNodes[i].presentState = 0
        return
    
    def checkNodesToVariableNodes(self): 
        # We go over the check nodes one by one (possibly in parallel in the future) and calculate outgoing values.
        # Then whenever we have a check node that completed calculating outgoing values, have it broadcast these value to the corresponding nodes.
        for i in range(self.numberOfCheckNodes):
            self.checkNodes[i].calcOutgoingValues()
            outgoingValues = self.checkNodes[i].outgoingValues
            #print(outgoingValues)
            for k in range(len(outgoingValues[:,0])):
                self.variableNodes[LDPC_INT_DATA_TYPE(outgoingValues[k,0])].update(outgoingValues[k,1])
            #for j in self.checkNodeAddressBook[i]:
                #a = self.checkNodes[i].getValueForNodeDictionary(j)
                #b = self.checkNodes[i].getValueForNode(j)
                #assert(a == b)
                #self.variableNodes[j].update(b)
            #    self.variableNodes[j].update(self.checkNodes[i].getValueForNodeDictionary(j))

        return

    def decoderStep(self):
        
        # If self.syndromeDecoding is True, then we start with a check2variable step first, and a decoder step is a broadcast from checknodes, followed by a return broadcast from check nodes to variable nodes
        # Otherwise we start with a variable2check step first.
        if self.syndromeDecoding:
            self.checkNodesToVariableNodes()
            self.variableNodesToCheckNodes()
        else:
            self.variableNodesToCheckNodes()
            self.checkNodesToVariableNodes()    
        
        # Reset the value of softVector
        softVector = self.softVector * 0
        # Finally we add the information from the channel to the present state and gather all present state values into a vector.
        for i in range(self.numberOfVariableNodes):
            self.variableNodes[i].presentState = self.variableNodes[i].presentState + self.variableNodes[i].fromChannel
            softVector[i] = self.variableNodes[i].presentState
        return softVector
        
    def decoderMainLoop(self, fromChannel, maxNumberOfIterations):
        status, binaryVector = self.isCodeword(fromChannel)
        softVector = np.copy(fromChannel)
        i = 0
        if status == 'Not a codeword':
            self.decoderSet(fromChannel)
            while (i < maxNumberOfIterations) & (status == 'Not a codeword'):
                i = i + 1
                softVector = self.decoderStep()
                status, binaryVector = self.isCodeword(softVector)
                #print('At iteration %d the status is: %s'%(i, status))
        return status, binaryVector, softVector, i



def main():
    print("*** In ldpc.py main function.")
    pass


# Omer Sella: Name guarding is needed when doing concurrent futures in Windows (i.e.: if __name__ ...)
if __name__ == '__main__':
    main()


