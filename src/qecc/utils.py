import numpy as np
from qecc.minSum import ldpcDecoder
from qecc.memBP import decode
import time
from qecc.polynomialCodes import A1_HX, A1_HZ
LDPC_INT_DATA_TYPE = np.int32

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


if __name__ == "__main__":
    numberOfTransmissions = 20
    seed = 123456
    errorRange = np.linspace(0.001, 0.1, 10)
    numberOfIterations = 50
    H = A1_HX.astype(np.int32)
    minSumEvaluateCode(numberOfTransmissions, seed, errorRange, numberOfIterations, H)
    memBPEvaluateCode(numberOfTransmissions, seed, errorRange, numberOfIterations, H)
    