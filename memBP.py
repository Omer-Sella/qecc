"""
The code in this file follows the explanations in "Improved belief propagation is sufficient for real-time decoding of quantum memory" https://arxiv.org/pdf/2506.01779
which points to the book "Modern Coding Theory" by Tom Richardson and Ruediger Urbanke https://www.cambridge.org/core/books/modern-coding-theory/A08C3B7B15351BF9C956CDFE5BE4846B

"""
import numpy as np
import copy

def extendedSign(x):
    """
    Extended sign function that returns 1 for x >= 0, -1 for x < 0
    """
    if x >= 0:
        return 1
    elif x < 0:
        return -1

    
def decoderInit(H, errorProbabilities, standardBP = True):
    """
    inputs:
    H: parity check matrix
    errorProbabilities: list of error probabilities for each qubit
    Notes
    -----
    Initialization and stopping: We initialize by setting the
    initial beliefs and biases as the log-likelihoods of the error priors Λj(0) = \ni_{j→i}(0) = log (1−pj)/pj
    and set the initial marginals from user input Mj (0) = Mj
    Bias term: Under standard BP, the bias factors are
    fixed, and the initial marginals are chosen as Mj = Λj(0),
    resulting in Λj(t) = Λj(0) for all t. In DMem-BP different
    initial marginals are allowed and the biases are
    updated via the equation
    """

    m,n = H.shape
    ni = copy.copy(H).astype(float)
    
    #print(ni)
    marginals = np.zeros(n)
    #for j in range(n):
    #    marginals[j] = initMarginals[j]
    Lambda = np.zeros(n)
    for j in range(n):
        Lambda[j] = np.log((1 - errorProbabilities[j]) / errorProbabilities[j])
        ni[:,j] *= np.log((1 - errorProbabilities[j]) / errorProbabilities[j])
        #for i in range(m):
            #print(np.log((1 - errorProbabilities[j]) / errorProbabilities[j]))
            #ni[i,j] *= np.log((1 - errorProbabilities[j]) / errorProbabilities[j])
            #print(f"ni[{i},{j}] = {ni[i,j]}")
        if standardBP:
            # set marginals to be the same as Lambda
            marginals[j] = np.log((1 - errorProbabilities[j]) / errorProbabilities[j])
    checkNeighbourhoods = {}
    errorNeighbourhoods = {}
    for i in range(m):
        checkNeighbourhoods[i] = np.where(H[i,:] == 1)[0].tolist()
    for j in range(n):
        errorNeighbourhoods[j] = np.where(H[:,j] == 1)[0].tolist()
    return marginals, Lambda, errorNeighbourhoods, checkNeighbourhoods, ni

def updateBiases(Lambda0, marginals, initMarginals, Gamma):
    Lambda = np.zeros(Lambda0.shape)
    for j in range(len(Lambda0)):
        Lambda[j] = (1-Gamma[j]) * Lambda0[j] + Gamma[j] * marginals[j]
    return Lambda


def calculateMarginalsAndHardDecision(Lambda, marginals, errorNeighbourhoods, mu):
    """
    Implements equation (3) in https://arxiv.org/pdf/2506.01779
    """
    errorVector = np.zeros(len(marginals), dtype=int)
    for j in range(len(marginals)):
        marginals[j] = Lambda[j] + np.sum([mu[l,j] for l in errorNeighbourhoods[j]])
        errorVector[j] = 0.5 * (1 - extendedSign(marginals[j]))
    return errorVector, marginals

def checkToErrorMessage( sigma, neighbourhoodOfCheckI, mu, ni, checkI, errorNodeJ):
    """
    Implements equation (1) in https://arxiv.org/pdf/2506.01779
    Check-to-error messages. updates mu[i,j] in place
    inputs:
    mu: check-to-error messages matrix - this is the thing we are updating (in place)
    sigma: syndrome vector
    Nchecks: dictionary of lists, where Nchecks[i] is the list of error nodes connected to check node i
    ni: error-to-check messages matrix
    i: check node index
    j: error node index
    """
    # We are computing a message from check node i to error node j, so we ignore the information from error node j
    neighbourhoodOfCheckI.remove(errorNodeJ)
    # The update to mu[i,j] is: µi→j = κi,j (−1)**σi * min_{ l∈N(i)\{j}} |ni[l,i]|
    # where κi,j = sign( ∏_{ l∈N(i)\{j}} ni[l,i] )
    kappaij = extendedSign(np.prod(ni[checkI, neighbourhoodOfCheckI]))
    #print("kappaij == {kappaij}")
    #print([np.abs(ni[Nchecksi,i]) for l in ni[i] if l != j])
    calc = kappaij * ((-1) ** sigma[checkI]) * np.min(np.abs(ni[checkI,neighbourhoodOfCheckI]))
    #print(calc)
    mu[checkI,errorNodeJ] = calc
    
    return

def errorToCheckMessage(Lambda, neighbourhoodOfErrorJ, mu, ni, checkI, errorNodeJ): 
    """
    Implements equation (2) in https://arxiv.org/pdf/2506.01779
    """
    neighbourhoodOfErrorJ.remove(checkI)
    ni[checkI,errorNodeJ] = Lambda[errorNodeJ] + np.sum(mu[neighbourhoodOfErrorJ,errorNodeJ])
    return


def decoderStep(checkNeighbourhoods, errorNeighbourhoods, ni, mu, Lambda, sigma):
    # In the implementation of the paper "Improved belief propagation is sufficient for real-time decoding of quantum memory" we first send an update from the check nodes to the error nodes
    # Go over all the check nodes
    for checkI in checkNeighbourhoods.keys():
        # For every error node connected to check node checkI, compute a message
        for errorNodeJ in checkNeighbourhoods[checkI]:
           # Compute the message from check node checkI to error node errorNodeJ
            checkToErrorMessage(sigma, copy.copy(checkNeighbourhoods[checkI]), mu, ni, checkI, errorNodeJ)
    # Go over all the error nodes
    for errorNodeJ in errorNeighbourhoods.keys():
        # For every check node connected to error node errorNodeJ, compute a message
        for checkI in errorNeighbourhoods[errorNodeJ]:
            # Compute the message from error node errorNodeJ to check node checkI
            errorToCheckMessage(Lambda, copy.copy(errorNeighbourhoods[errorNodeJ]), mu, ni, checkI, errorNodeJ)
    return

def decode(H, sigma, initMarginals, errorProbabilities, Gammas, maxIterations=10):
    marginals, Lambda, errorNeighbourhoods, checkNeighbourhoods, ni = decoderInit(H, errorProbabilities)
    mu = np.zeros(H.shape)
    converged = False
    for iteration in range(maxIterations):
        decoderStep(checkNeighbourhoods, errorNeighbourhoods, ni, mu, Lambda, sigma)
        errorVector, marginals = calculateMarginalsAndHardDecision(Lambda, marginals, errorNeighbourhoods, mu)
        if np.all(np.mod(H @ errorVector, 2) == sigma):
            converged = True
            break
    return errorVector, marginals, converged, iteration



if __name__ == "__main__":
    pass

