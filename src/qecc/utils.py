import numpy as np
from qecc.gf4 import integerToDualBinary
from qecc.logicals import computeLogicals
from qecc.minSum import ldpcDecoder
from qecc.memBP import decode
import time
from qecc.polynomialCodes import A1_HX, A1_HZ
LDPC_INT_DATA_TYPE = np.int32
FLOAT_DATA_TYPE_UTILS = np.float32

def minSumEvaluateCode(numberOfTransmissions, seed, errorRange, numberOfIterations, H):
    """
    parameters
    ----------
    numberOfTransmissions : int
        Number of transmissions to simulate at each error probability.
    seed : int
        Seed for the local PRNG.
    errorRange : list of float
        List of error probabilities to simulate.
    numberOfIterations : int
        Number of iterations for the decoder.
    H : ndarray
        Parity check matrix.
    Returns
    ------- 
    berArray : ndarray
        Array of bit error rates for each error probability in errorRange.

    Notes
    -----
    Concurrent futures require the seed to be between 0 and 2**32 -1
    """
    codewordSize = H.shape[1]
    localPrng = np.random.RandomState(seed)
    decoder = ldpcDecoder(H, syndromeDecoding = True)
    start = 0
    end = 0
    codeword = np.zeros(codewordSize, dtype = LDPC_INT_DATA_TYPE)
    decodedWord = np.zeros(codewordSize, dtype = LDPC_INT_DATA_TYPE)
    berArray = np.zeros(len(errorRange))
    for i,p in zip(range(len(errorRange)), errorRange):
        timeTotal = 0    
        print(f"Error prob {p}, corresponding to i=={i}")
        for k in range(numberOfTransmissions):
            error = localPrng.choice([0,1], size=codewordSize, replace=True, p=[1 - p, p])
            errorModulated = np.where(error == 0, -1.0, 1.0)
            berUncoded = 0
            berDecoded = 0
            start = time.time()
            status, decodedWord, softVector, iterationStoppedAt = decoder.decoderMainLoop(errorModulated, numberOfIterations)
            end = time.time()
            timeTotal += (end - start)
            #print("******** " + str(np.sum(decodedWord == codeword)))
            berDecoded = np.count_nonzero(decodedWord != codeword)
            berArray[i] += berDecoded
        print("Time it took the decoder:")
        print(timeTotal)
        print("And the throughput is:")
        numberOfBits = numberOfTransmissions * codewordSize
        print(numberOfBits / timeTotal)
    return berArray / (numberOfTransmissions * codewordSize)

def memBPEvaluateCode(numberOfTransmissions, seed, errorRange, numberOfIterations, H):
    """
    parameters
    ----------
    numberOfTransmissions : int
        Number of transmissions to simulate at each error probability.
    seed : int
        Seed for the local PRNG.
    errorRange : list of float
        List of error probabilities to simulate.
    numberOfIterations : int
        Number of iterations for the decoder.
    H : ndarray
        Parity check matrix.
    Returns
    ------- 
    berArray : ndarray
        Array of bit error rates for each error probability in errorRange.

    Notes
    -----
    Concurrent futures require the seed to be between 0 and 2**32 -1
    """
    codewordSize = H.shape[1]
    localPrng = np.random.RandomState(seed)
    start = 0
    end = 0
    codeword = np.zeros(codewordSize, dtype = np.int32)
    decodedWord = np.zeros(codewordSize, dtype = np.int32)
    berArray = np.zeros(len(errorRange))
    for i,p in zip(range(len(errorRange)), errorRange):
        timeTotal = 0    
        print(f"Error prob {p}, corresponding to i=={i}")
        for k in range(numberOfTransmissions):
            error = localPrng.choice([0,1], size=codewordSize, replace=True, p=[1 - p, p])
            start = time.time()
            errorVector, marginals, converged, iteration = decode(H, initMarginals = None, errorProbabilities= [p]*codewordSize, sigma = (H @ error %2), Gammas = None, maxIterations=numberOfIterations, logProbabilities = False)
            end = time.time()
            timeTotal += (end - start)
            berDecoded = np.count_nonzero(errorVector != error)
            berArray[i] += berDecoded
        print("Time it took the decoder:")
        print(timeTotal)
        print("And the throughput is:")
        numberOfBits = numberOfTransmissions * codewordSize
        print(numberOfBits / timeTotal)
    return berArray / (numberOfTransmissions * codewordSize)

def wrapperForRoffesLdpc(H, syndrome, initialValues, decoderStoppingCriterion):
    from ldpc import bposd_decoder
    p_error = np.average(initialValues[:,1]) # initialValues[i,1] is the probability of error for the ith coordinate
    bpDecoder=BpOsdDecoder(H,#the parity check matrix
        error_rate=p_error,
        channel_probs= initialValues[:,1], #assign error_rate to each qubit. This will override "error_rate" input variable
        max_iter=decoderStoppingCriterion, #the maximum number of iterations for BP)
        bp_method="ms",
        ms_scaling_factor=0, #min sum scaling factor. If set to zero the variable scaling factor method is used
        osd_method="osd0", #the OSD method. Choose from:  1) "osd_e", "osd_cs", "osd0"
        osd_order=0 #the osd search depth
        )
    result = bpDecoder.decode(syndrome)
    return result, True

def decoderEvaluator(decoderFunction, dualBinary, Hx, Hz, errorRange, decoderStoppingCriterion, numberOfSamples):
    """
    Arguments:
    Hx, Hz: A pair of binary matrices. Use the codes in polynomialCodes.py
    errorRange: A range of probabilities
    decoderStoppingCriterion - usually intended as maximum number of iterations.

    Returns:
    logicalErrorRate: array of floats. For each error probability p, the number of times a logical error was created divided by the number of samples attempted for the error probability.
    decoderFailureRate: array of floats. For each error probability p, the number of times the decoder failed divided by the number of samples attempted for the error probability.
    """
    import numpy as np
    from qecc.gf4 import integerTraceProduct as tp
    from qecc.gf4 import integerToDualBinary, binaryDualToInteger
    from qecc.logicals import computeLogicals
    seed = 7134066
    localRandom = np.random.RandomState(seed)
    logicalX, logicalZ = computeLogicals(Hx, Hz)

    decoderFailureRate = {}
    logicalErrorRate = {}
    for p in errorRange:
        decoderFailureRate[p] = 0
        logicalErrorRate[p] = 0
        for _ in range(numberOfSamples):
            #Sample (some number of times) an error, which is a vector over {0,1,2,3} representing I,X,Z,Y (to be consistent check with the documentation in gf4.py)
            error = localRandom.choice([0,1,2,3], size=Hx.shape[1], replace=True, p=[1 - 3*p, p, p, p])
            errorX, errorZ = integerToDualBinary(error)
            #Calculate the syndrome for this error
            syndromeX, syndromeZ = Hx.dot(errorZ)%2, Hz.dot(errorX)%2
            #Run whatever decoder you decided to evaluate with arguments Hx, Hz, syndrome. The decoder has to accept a binary pair and syndrome, and return errorEstimation (an estimated solution), which is an array over 0,1,2,3 and a success flag (boolean).
            
            if dualBinary:
                
                initialValuesX = np.tile(np.array([1-2*p,2 * p]), (Hx.shape[1], 1))
                initialValuesZ = np.tile(np.array([1-2*p,2 * p]), (Hz.shape[1], 1))
                estimatedErrorX, estimatedErrorZ, success = decoderFunction(Hx, Hz, syndromeX, syndromeZ, initialValuesX, initialValuesZ, decoderStoppingCriterion)
            else:
                initialValues = np.tile(np.array([1-3*p, p, p, p]), (Hx.shape[1], 1))
                syndrome = np.hstack((syndromeX, syndromeZ))
                H = np.vstack((Hx, 2*Hz))
                estimatedError, success = decoderFunction(H, syndrome, initialValues, decoderStoppingCriterion)
                estimatedErrorX, estimatedErrorZ = integerToDualBinary(estimatedError)
            #If the decoder thinks it failed, add 1 to the decoderFailure counter. 
            if not success:
                decoderFailureRate[p] = decoderFailureRate[p] + 1
            else:

                #If the decoder thinks it succeeded (success == True), then test the residual error, i.e.: the (gf(4)) sum of the estimated error and the original error, to see if either: 
                #It does not amount to a 0 syndrome (this means the decoder is wrong, meaning - it thinks that the estimatedError + error is a stabilizer, and that it countered the effect of the error up to a stabilizer, but it didn't, and the result is not in the normalizer).
                #It is a logical error.
                #In either case we increase the decoderErrorRate by 1.
                residualErrorX = (estimatedErrorX + errorX) % 2
                residualErrorZ = (estimatedErrorZ + errorZ) % 2
                # Check whether the residual error gives 0 syndrome:
                if not ( np.all((np.dot(Hx,residualErrorZ)) % 2 ==0) and np.all((np.dot(Hz, residualErrorX)%2) == 0)):
                    print("Decoder failure: the residual error does not give 0 syndrome, meaning the decoder is wrong")
                    logicalErrorRate[p] += 1
                else: # So we are in the case that the residual error commutes with all stabilizers, i.e., it is in the normalizer. So let's check if it is a stabilizer (commutes with all logicals), or a logical error (anticommutes with some logical operator)
                    if not ( np.all((np.dot(logicalZ,residualErrorX) % 2 )== 0) and np.all((np.dot(logicalX, residualErrorZ) % 2)==0)):
                        print(f"Logical error: the residual error commutes with all stabilizers but anticommutes with some logical operator")
                        logicalErrorRate[p] += 1
    return logicalErrorRate, decoderFailureRate


def binaryDecoderToDualBinaryDecoderWrapper(binaryDecoderFunction):
    def dualDecoder(Hx, Hz, syndromeX, syndromeZ, initialValuesX, initialValuesZ, decoderStoppingCriterion):
        estimatedErrorZ, successZ = binaryDecoderFunction(Hx, syndromeX, initialValuesX,decoderStoppingCriterion)
        estimatedErrorX, successX = binaryDecoderFunction(Hz, syndromeZ, initialValuesZ,decoderStoppingCriterion)
        return estimatedErrorX, estimatedErrorZ, successX and successZ
    return dualDecoder



if __name__ == "__main__":
    
    
    from qecc.qbp import refinedBPalgorithm3
    from qecc.polynomialCodes import A1_HX, A1_HZ
    from qecc.minSum import ldpcDecoderWrapper

    # Check quaternary BP is working
    #logicalER, decoderFailureRate = decoderEvaluator(decoderFunction = refinedBPalgorithm3, dualBinary = False, Hx = A1_HX, Hz = A1_HZ, errorRange = [0.01, 0.001, 0.0001, 0.00001], decoderStoppingCriterion = 20, numberOfSamples = 10)
    #print(f"Logical error rate: {logicalER}")
    #print(f"Decoder failure rate: {decoderFailureRate}")
    
    logicalER, decoderFailureRate = decoderEvaluator(decoderFunction = ldpcDecoderWrapper, dualBinary = True, Hx = A1_HX, Hz = A1_HZ, errorRange = [0.01, 0.001, 0.0001, 0.00001], decoderStoppingCriterion = 20, numberOfSamples = 10)
    print(f"Logical error rate: {logicalER}")
    print(f"Decoder failure rate: {decoderFailureRate}")
    dualRoffeDecoder = binaryDecoderToDualBinaryDecoderWrapper(wrapperForRoffesLdpc)
    logicalER, decoderFailureRate = decoderEvaluator(decoderFunction = dualRoffeDecoder, dualBinary = True, Hx = A1_HX, Hz = A1_HZ, errorRange = [0.01, 0.001, 0.0001, 0.00001], decoderStoppingCriterion = 20, numberOfSamples = 10)
    print(f"Logical error rate: {logicalER}")
    print(f"Decoder failure rate: {decoderFailureRate}")
    # numberOfTransmissions = 20
    # seed = 123456
    # errorRange = np.linspace(0.001, 0.1, 10)
    # numberOfIterations = 50
    # H = A1_HX.astype(np.int32)
    # minSumEvaluateCode(numberOfTransmissions, seed, errorRange, numberOfIterations, H)
    # memBPEvaluateCode(numberOfTransmissions, seed, errorRange, numberOfIterations, H)
    