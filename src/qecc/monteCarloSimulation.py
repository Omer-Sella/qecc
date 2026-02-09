# So the conclusion is that we need reliabilities that are as accurate as possible
import argparse
import qecc.polynomialCodes as polynomialCodes
from qecc.memBP import decode
from qecc.funWithMatrices import binaryGaussianEliminationOnRows
from qecc import funWithMatrices, logicals
from qecc.logicals import computeLogicals
import copy
import matplotlib.pyplot as plt
import numpy as np
from qecc.osd import osdDecoder


def monteCarloSimulation(Hx, Hz, osdDecode = True, numberOfSamples = 100, pErrorList = np.linspace(0.001, 0.1, 10), maxIterations =50, seed = 777):
    if Hx in polynomialCodes.codes.values():
        Hx = copy.copy(Hx.astype(np.int32))
    if Hz in polynomialCodes.codes.values():
        Hz = copy.copy(Hz.astype(np.int32))
    localRandom = np.random.RandomState(seed)
    L_X, L_Z = logicals.computeLogicals(Hx, Hz)
    logicalErrors = np.zeros(len(pErrorList))
    ber = np.zeros(len(pErrorList))
    for p in range(len(pErrorList)):
        probabilityOfError = pErrorList[p]
        print(f"Simulating for probability of error {probabilityOfError}")
        for i in range(numberOfSamples):
            error = localRandom.random.choice([0,1], size=(Hx.shape[1],), p=[1-probabilityOfError, probabilityOfError]).astype(np.int32)
            coordinateReliabilities = np.ones(Hx.shape[1]) * probabilityOfError
            initMarginals = np.ones(Hx.shape[1]) * np.log( (1 - probabilityOfError) / probabilityOfError )
            errorVector, marginals, converged, iteration = decode(Hx, initMarginals = initMarginals, errorProbabilities= initMarginals, sigma = (Hx @ error %2), Gammas = None, maxIterations=maxIterations, logProbabilities = True)
            #Count bit errors
            ber[p]+= np.sum(np.where(errorVector  == error, 0, 1))
            if osdDecode and not converged:
                solution, reliability = osdDecoder(Hx, (Hx @ error %2), marginals)
                # Check whether a logical error occurred
                residualError = (error + solution) %2
                if np.any(Hx @ residualError % 2) or np.any(L_X @ residualError % 2):
                    logicalErrors[p] += 1
    return logicalErrors, ber


if __name__ == "__main__":
    argumentParser = argparse.ArgumentParser(description="Monte Carlo simulation of OSD decoding on a given code.")
    argumentParser.add_argument("--hx", type=str, default="A1_HX", help="H_X matrix to use.")
    argumentParser.add_argument("--hz", type=str, default="A1_HZ", help="H_Z matrix to use.")
    argumentParser.add_argument("--samples", type=int, default=100, help="Number of samples to simulate.")
    argumentParser.add_argument("--osd", action="store_true", help="Whether to use OSD decoding.")
    argumentParser.add_argument("--maxIterations", type=int, default=50, help="Maximum number of BP iterations.")
    argumentParser.add_argument("--seed", type=int, default=777, help="Random seed.")
    args = argumentParser.parse_args()