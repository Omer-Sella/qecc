# surrogate/generateProbeEvaluations.py
"""Generate labelled probe evaluations at a given (l, m) for transfer validation.

Uses bb_gym_v_0_1's own BpOsdDecoder evaluation (imported, not modified) so
probe labels are produced by the exact pipeline that produced the training
logs. Records are written in the canonical codeEvaluations_*.jsonl schema.

Usage:
    python surrogate/generateProbeEvaluations.py --l 9 --m 9 --budget 1000 \
        --output-directory C:/Users/Omer/rl-qecc-data/probes_9x9 --seed 1
"""
import argparse
import os

import numpy as np

from qecc.codeEvaluationDataset import CANONICAL_ERROR_RANGE
from qecc.logicals import calculateCodeDimension
from qecc.polynomialCodes import generateABmatrices, bicycleCodeFromAB


def generateProbes(l, m, budget, outputDirectory, seed,
                   minimumK=6, numberOfSamples=50):
    os.makedirs(outputDirectory, exist_ok=True)
    # The env's logger reads QECC_DATA at call time; point it at our output dir
    # for this process only.
    os.environ["QECC_DATA"] = outputDirectory
    from qecc.bb_gym_v_0_1 import bicycleBivariateCodeEnvironment
    environment = bicycleBivariateCodeEnvironment(
        l, m, errorRange=CANONICAL_ERROR_RANGE, numberOfSamples=numberOfSamples,
        minimumNumberOfLogicalQubits=minimumK, seed=seed * 100000,
        codeLogging=True, bitFlipping=True)
    randomGenerator = np.random.default_rng(seed)
    evaluated = 0
    attempts = 0
    while evaluated < budget and attempts < budget * 20:
        attempts += 1
        environment.aX = randomGenerator.integers(0, 2, size=l).astype(np.int16)
        environment.aY = randomGenerator.integers(0, 2, size=m).astype(np.int16)
        environment.bX = randomGenerator.integers(0, 2, size=l).astype(np.int16)
        environment.bY = randomGenerator.integers(0, 2, size=m).astype(np.int16)
        environment.A, environment.B = generateABmatrices(
            l, m,
            np.where(environment.aX != 0)[0], np.where(environment.aY != 0)[0],
            np.where(environment.bX != 0)[0], np.where(environment.bY != 0)[0])
        environment.Hx, environment.Hz = bicycleCodeFromAB(environment.A, environment.B)
        k = calculateCodeDimension(environment.Hx, environment.Hz)
        if k < minimumK:
            continue
        environment.numberOfLogicalQubits = k
        environment.seed = environment.seed + 1
        environment.decoderEvaluation(environment.seed)  # writes the jsonl record
        evaluated += 1
        if evaluated % 25 == 0:
            print(f"evaluated {evaluated}/{budget} (attempts {attempts})", flush=True)
    print(f"done: evaluated {evaluated} codes in {attempts} attempts")
    return evaluated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--l", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--minimum-k", type=int, default=6)
    parser.add_argument("--number-of-samples", type=int, default=50)
    arguments = parser.parse_args()
    generateProbes(arguments.l, arguments.m, arguments.budget,
                   arguments.output_directory, arguments.seed,
                   arguments.minimum_k, arguments.number_of_samples)


if __name__ == "__main__":
    main()
