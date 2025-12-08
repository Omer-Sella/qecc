import pathlib
import os

projectDir = os.environ.get('QECC')
if projectDir == None:
    raise Exception("Please set the environment variable 'QECC' to point to the root directory of the project.")
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.append(projectDir)
from src.memBP import decode, errorToCheckMessage, checkToErrorMessage, decoderInit, calculateMarginalsAndHardDecision
from scipy import sparse
import numpy as np
import copy



LDPC_INT_DATA_TYPE = np.int32

def test_decoder_1_bit_flip():
    from polynomialCodes import A4_HX as H
    success = True
    for i in range(H.shape[1]):
        error = np.zeros(H.shape[1], dtype=bool)
        error[i] = 1
        sindromes = (H @ error) % 2
        errorProbabilities = [0.01] * H.shape[1]
        Gammas = None
        decodedError, marginals,_,_ = decode(H, sindromes, None, errorProbabilities, Gammas, maxIterations=10)
        if not np.all(error == decodedError):
            success = False
    if success:
        return 'OK'
    else:
        return 'FAIL'


def test_decoder_k_bit_flips(k=2):
    """
    Note that this test is failing for A4_HX. I haven't tested it using an alternative decoder, but for now I am satisfied that this is consistent.
    """
    from polynomialCodes import A4_HX as H
    index1, index2 = np.random.choice(H.shape[1], 2, replace=False)
    success = True
    for i in range(1):#H.shape[1]):
        error = np.zeros(H.shape[1], dtype=float)
        error[index1] = 1
        error[index2] = 1
        sindromes = (H @ error) % 2
        errorProbabilities = [1/H.shape[1]] * H.shape[1]
        Gammas = None
        decodedError, marginals, converged, iterations = decode(H, sindromes, None, errorProbabilities, Gammas, maxIterations=10)
        print(error)
        print(decodedError)
        #print(marginals)
        print(f"Converged: {converged} after {iterations} iterations")
        assert(np.all(error == decodedError))
    return 'OK'


def test_15_1():
    H = np.array([[1,1,1,0,0,1,1,0,0,1],
                  [1,0,1,0,1,1,0,1,1,0],
                  [0,0,1,1,1,0,1,0,1,1],
                  [0,1,0,1,1,1,0,1,0,1], 
                  [1,1,0,1,0,0,1,1,1,0]])
    
    nTest=np.array([[-1.3,-1.7,-1.5, 0   , 0  , 1.9, -1.5,   0 ,    0, 1.2],
                    [-1.3,0   ,-1.5, 0   , 0.2, 1.9,    0, 1.3 , -1.1, 0  ],
                    [0   ,0   ,-1.5,-0.08, 0.2, 0  , -1.5,   0 , -1.1, 1.2],
                    [0   ,-1.7,0   ,-0.08, 0.2, 1.9,    0, 1.3 ,    0, 1.2], 
                    [-1.3,-1.7,0   ,-0.08, 0  , 0  , -1.5, 1.3 , -1.1, 0  ]])
    
    
    errorProbabilities = [0.22, 0.16, 0.19, 0.48, 0.55, 0.87, 0.18, 0.79, 0.25, 0.76]
    #errorProbabilities = [0.2] * 10
    codeword = [0,0,0,1,0,1,0,1,0,1]
    received = [0,0,0,0,1,1,0,1,0,1]
    error =    [0,0,0,1,1,0,0,0,0,0]
    sigma = H @ np.array(error) % 2
    assert (error == (np.array(codeword) + np.array(received)) % 2).all()
    mu = np.zeros(H.shape, dtype=float)
    marginals, Lambda, errorNeighbourhoods, checkNeighbourhoods, ni = decoderInit(H, None, errorProbabilities, standardBP= True, logProbabilities= False)
    for i in range(len(errorProbabilities)):
        assert(marginals[i] == np.log((1 - errorProbabilities[i])/errorProbabilities[i]))
    
    
    assert(np.all(checkNeighbourhoods[0] == [0,1,2,5,6,9]))
    assert(np.all(checkNeighbourhoods[1] == [0,2,4,5,7,8]))
    assert(np.all(checkNeighbourhoods[2] == [2,3,4,6,8,9]))
    assert(np.all(checkNeighbourhoods[3] == [1,3,4,5,7,9]))
    assert(np.all(checkNeighbourhoods[4] == [0,1,3,6,7,8]))

    epsilon = 0.1 # Tolerance for difference between T.K. Moon's 15.7 and our implementation
    assert np.all(np.abs(( np.array(marginals) - np.array([1.3, 1.7, 1.5, 0.08, -0.2, -1.9, 1.5, -1.3, 1.1, -1.2]))) < epsilon) # Not that there is a change of sign, as the example in T.K. Moon is log( prob(something == 1) / prob(something == 0)) and IBM chose to invert it.
    assert np.all(np.abs(ni + nTest) < epsilon) # Notice + instead of - because of the sign change mentioned above.
    # # Now let's have a first message passing from error (variable) nodes to check nodes, UNLIKE the order in decoder step:

    for errorNodeJ in errorNeighbourhoods.keys():
        # For every check node connected to error node errorNodeJ, compute a message
        for checkI in errorNeighbourhoods[errorNodeJ]:
            # Compute the message from error node errorNodeJ to check node checkI
            errorToCheckMessage(Lambda, copy.copy(errorNeighbourhoods[errorNodeJ]), mu=mu, ni=ni, checkI=checkI, errorNodeJ=errorNodeJ)

def evaluateCode(numberOfTransmissions, seed, SNRpoints, numberOfIterations, H):
    """
    Decoder wrapper to evaluate a code over a range of SNR points and parity matrix H over the AWGN channel.
    """
    
    # Concurrent futures require the seed to be between 0 and 2**32 -1
    #assert (np.dtype(seed) == np.int32)
    assert (seed > 0)
    assert hasattr(SNRpoints, "__len__")
    localPrng = np.random.RandomState(seed)
    numberOfSNRpoints = len(SNRpoints)
    start = 0
    end = 0
    
    codeword = np.zeros(H.shape[1], dtype = LDPC_INT_DATA_TYPE)
    decodedWord = np.zeros(H.shape[1], dtype = LDPC_INT_DATA_TYPE)
    modulatedCodeword = np.where(codeword == 0 , 1, -1)
    berArray = np.zeros(numberOfSNRpoints, dtype = np.float32)
    for s in range(numberOfSNRpoints):
        timeTotal = 0    
        SNR = SNRpoints[s]
        berDecoded = 0
        for k in range(numberOfTransmissions):
            ## Now use the definition: SNR = signal^2 / sigma^2
            sigma = np.sqrt(0.5 / (10 ** (SNR/10)))
            #print(sigma)
            noise = localPrng.normal(0, sigma, H.shape[1])
            noisyModulated = modulatedCodeword + noise
            recievedWord = np.where(noisyModulated >= 0, 0, 1)            
            errorVector, marginals, converged, iteration = decode(H, initMarginals = noisyModulated, errorProbabilities= noisyModulated, sigma = np.zeros(H.shape[0]), Gammas = None, maxIterations=numberOfIterations, logProbabilities = True)
            decodedWord = errorVector # May seem redundant, but it's here to remond that in this test, we are using memBP as a decoder that attempts to finds a codeword, not an errorVector, so the errorVector returned is actually the codeword
            berDecoded += np.count_nonzero(decodedWord != codeword)
        # print(s)
        # print(berArray)
        # print(berArray[s])
        berArray[s] = berDecoded
    numberOfBits = numberOfTransmissions * H.shape[1]
    return berArray / numberOfBits

def test_nearEarthAt_3_6():

    """
    This test is supposed to culminate in 0 errors
    Good for debug.
    """
    import time
    import pathlib
    nearEarthPath = str(projectDir).replace("\\","/") + "/codeMatrices/nearEarthParityCsr.npz"
    nearEarthParity = sparse.load_npz(nearEarthPath).toarray()
    #numOfTransmissions = 50
    SNRRegionOfInterest = [3.6]
    codewordSize = 8176
    messageSize = 7154
    numOfIterations = 50
    numOfTransmissions = 50
    start = time.time()
    bStats = evaluateCode(numOfTransmissions, 460101, SNRRegionOfInterest, numOfIterations, nearEarthParity)   
    end = time.time()

    # Expected result for SNR == [3.0, 3.2, 3.4,3.6] at 50 iterations: [0.01953278 0.00419521 0.         0.        ]
    epsilon = 0.00001
    assert(bStats[0] < epsilon)

def test_NearEarth(numOfTransmissions = 2):
    #print("*** in test near earth")
    import time
    import pathlib
    nearEarthPath = str(projectDir).replace("\\","/") + "/codeMatrices/nearEarthParityCsr.npz"
    nearEarthParity = sparse.load_npz(nearEarthPath).toarray()
    SNRRegionOfInterest = [3.0, 3.2, 3.4,3.6]
    codewordSize = 8176
    messageSize = 7154
    numOfIterations = 10
    start = time.time()
    bStats = evaluateCode(numOfTransmissions, 460101, SNRRegionOfInterest, numOfIterations, nearEarthParity)    
    end = time.time()
    # Expected result for SNR == [3.0, 3.2, 3.4,3.6] at 50 iterations: [0.01953278 0.00419521 0.         0.        ]
    experimental = [0.01953278, 0.00419521, 0.0,         0.0        ]
    epsilon = [0.01, 0.01, 0.001, 0.0001]
    # print(f"BER stats : {bStats}")
    # print(f"{bStats - experimental}")
    # print(f"{np.abs(bStats - experimental) < epsilon}")
    assert(np.all(np.abs(bStats - experimental) < epsilon))



if __name__ == "__main__":
    test_decoder_1_bit_flip()
    #print(test_decoder_k_bit_flips(k=2))
    test_15_1()
    test_nearEarthAt_3_6()
    test_NearEarth()
    pass
 