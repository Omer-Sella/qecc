"""
The code in this file follows the explanations in "Improved belief propagation is sufficient for real-time decoding of quantum memory"
which points to the book "Modern Coding Theory" by Tom Richardson and Ruediger Urbanke https://www.cambridge.org/core/books/modern-coding-theory/A08C3B7B15351BF9C956CDFE5BE4846B

"""
import numpy as np


def decoderInit(H, errorProbabilities):
    """
    inputs:
    H: parity check matrix
    errorProbabilities: list of error probabilities for each qubit
    Notes
    -----
    Bias term: Under standard BP, the bias factors are
    fixed, and the initial marginals are chosen asMj = Λj(0),
    resulting in Λj(t) = Λj(0) for all t. In DMem-BP different
    initial marginals are allowed and the biases are
    updated via the equation
    """
    m,n = H.shape
    marginals = np.zeros(n)
    #for j in range(n):
    #    marginals[j] = initMarginals[j]
    Lambda0 = np.zeros(n)
    for i in range(n):
        Lambda0[i] = np.log((1 - errorProbabilities[i]) / errorProbabilities[i])
        marginals[i] = Lambda0[i]

    Nchecks = {}
    Nerrors = {}
    for i in range(m):
        Nchecks[i] = list(np.where(H[i,:] == 1))
    for j in range(n):
        Nerrors[j] = list(np.where(H[:,j] == 1))
    return marginals, Lambda0, Nerrors, Nchecks

def updateBiases(Lambda0, marginals, initMarginals, Gamma):
    Lambda = np.zeros(Lambda0.shape)
    for j in range(len(Lambda0)):
        Lambda[j] = (1-Gamma[j]) * Lambda0[j] + Gamma[j] * marginals[j]
    return Lambda


def calculateMarginalsAndHardDecision(Lambda, marginals, Nerrors, mu):
    errorVector = np.zeros(len(marginals), dtype=int)
    for j in range(n):
        marginals[j] = Lambda[j] + np.sum([mu[l] for l in Nerrors[j]])
        errorVector[j] = 0.5 * (1 - np.sign(marginals[j]))
    return errorVector,

def checkToErrorMessage(mu, sigma, Nchecks, ni, i, j):
    """
    Check-to-error messages. updates mu[i,j] in place
    inputs:
    mu: check-to-error messages matrix
    sigma: syndrome vector
    Nchecks: dictionary of lists, where Nchecks[i] is the list of error nodes connected to check node i
    ni: error-to-check messages matrix
    i: check node index
    j: error node index
    """
    kappaij = np.sign(np.prod([ni[l,i] for l in Nchecks[i] if l != j]))
    mu[i,j] = kappaij * ((-1) ** sigma[i]) * min([np.abs(ni[l,i]) for l in N[i] if l != j])
    return

def errorToCheckMessage(Lambda, Nerrors, mu, ni, j, i):
    ni[j,i] = Lambda[j] + np.sum([mu[l,i] for l in Nerrors[i] if l != j])
    return


def decoderStep(Nchecks, Nerrors, ni, mu, Lambda):
    for i in Nchecks.keys:
        for j in Nchecks[i]:
            checkToErrorMessage(mu, sigma, Nchecks[i], ni, i, j)
    for j in Nerrors.keys:
        for i in Nerrors[j]:
            errorToCheckMessage(Lambda, Nerrors[j], mu, ni, j, i)
    return

def decode(H, sigma, initMarginals, errorProbabilities, Gammas, maxIterations=10):
    marginals, Lambda0, Nerrors, Nchecks = decoderInit(H, errorProbabilities)
    Lambda = Lambda0
    m,n = H.shape
    ni = np.zeros((n,m))
    mu = np.zeros((m,n))
    for iteration in range(maxIterations):
        decoderStep(Nchecks, Nerrors, ni, mu, Lambda)
        errorVector, marginals = calculateMarginalsAndHardDecision(Lambda, marginals, Nerrors, mu)
        if np.all(np.mod(H @ errorVector, 2) == sigma):
            converged = True
            break
    return errorVector, marginals
