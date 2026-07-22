"""Post-training evaluation of a learned BB-code PPO policy.

A single, self-contained post-processing script. It reuses the project's existing
code construction, decoders, dataset loader and logging; nothing else in the tree
is modified. Design record:
docs/superpowers/specs/2026-07-16-policy-evaluation-script-design.md

Two decoupled stages
--------------------
Stage 1 (cheap, decoder effectively off): load a trained hybrid policy, roll it out
deterministically from the all-zero code and from the worst logged codes, and record
only the sequence of codes visited. Deterministic dynamics on a finite state space
guarantee the trajectory eventually cycles, so we stop at the first repeated state.

Stage 2 (the expensive, embarrassingly-parallel part): score every DISTINCT visited
code under a sweep of decoders (BP-OSD orders w=0..5 and plain quaternary BP), on both
noise grids, using a fixed-logical-failure-count sampling budget; add structural metrics
(n, k, rate, check weight) and, optionally, exact minimum distance by mixed-integer
programming. Report distinct-code equivalence classes and reference-code anchors.

Run in the background on an idle shared box; no GPU, no training.
"""

import argparse
import functools
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch

# ---- reused, never re-implemented ----
from qecc.modelArchitectures import hybridNet
from qecc.reinforcementLearning import ConcatenatedOneHotCategorical
from qecc.utils import (decoderEvaluator, wrapperForRoffesLdpc,
                        binaryDecoderToDualBinaryDecoderWrapper)
from qecc.bb_gym_v_0_1 import bicycleBivariateCodeEnvironment
from qecc.polynomialCodes import generateABmatrices, bicycleCodeFromAB, bbCodes
from qecc.logicals import calculateCodeDimension, computeLogicals
from qecc import funWithMatrices
from qecc.loggerForReinforcementLearning import logger
from qecc.codeEvaluationDataset import loadCodeEvaluations, rewardFromCounts

FIVE_POINT_GRID = np.linspace(1e-4, 1e-1, 5)
TEN_POINT_GRID = np.linspace(1e-4, 1e-1, 10)

# The published Bivariate Bicycle codes (Bravyi et al 2024, Table 3), keyed by (l, m),
# used as reference anchors. Polynomials as monomial-exponent lists, matching
# generateABmatrices(l, m, aX, aY, bX, bY).
REFERENCE_CODES = {
    (6, 6):  dict(name="[[72,12,6]]",   aX=[3],    aY=[1, 2],  bX=[1, 2],  bY=[3]),
    (15, 3): dict(name="[[90,8,10]]",   aX=[9],    aY=[1, 2],  bX=[0, 2],  bY=[7]),
    (9, 6):  dict(name="[[108,8,10]]",  aX=[3],    aY=[1, 2],  bX=[1, 2],  bY=[3]),
    (12, 6): dict(name="[[144,12,12]]", aX=[3],    aY=[1, 2],  bX=[1, 2],  bY=[3]),
}


# =====================================================================================
# Small shared helpers
# =====================================================================================

def gf2Rank(matrix):
    """GF(2) rank via the project's Gaussian elimination (reused, not re-implemented)."""
    if matrix.size == 0:
        return 0
    _, _, rank = funWithMatrices.binaryGaussianEliminationOnRows(np.array(matrix, copy=True))
    return int(rank)


def polynomialsToIndices(aX, bX, aY, bY):
    """Coefficient vectors (0/1) -> monomial-exponent index lists that generateABmatrices wants."""
    return (np.where(np.asarray(aX) != 0)[0],
            np.where(np.asarray(aY) != 0)[0],
            np.where(np.asarray(bX) != 0)[0],
            np.where(np.asarray(bY) != 0)[0])


def buildCode(l, m, aX, bX, aY, bY):
    """Build (Hx, Hz) from 0/1 coefficient vectors, via the same path the environment uses."""
    aXi, aYi, bXi, bYi = polynomialsToIndices(aX, bX, aY, bY)
    A, B = generateABmatrices(l, m, aXi, aYi, bXi, bYi)
    return bicycleCodeFromAB(A, B)


def checkWeight(aX, bX, aY, bY):
    """Weight of a BB check row = number of monomials across A and B (Bravyi codes: 6)."""
    return int(np.count_nonzero(aX) + np.count_nonzero(aY) +
               np.count_nonzero(bX) + np.count_nonzero(bY))


def rateToInverseInteger(k, n):
    """Rate k/n rounded DOWN to the nearest inverse integer 1/j (Bravyi et al convention)."""
    if k <= 0:
        return 0.0, None
    inverse = int(np.ceil(n / k))          # smallest j with 1/j <= k/n
    return 1.0 / inverse, inverse


# =====================================================================================
# Stage 1: crawl + policy loading + deterministic rollouts
# =====================================================================================

def parseExperimentComments(experimentPath):
    """Parse the '# key = value' header of an experiment.txt into {str: str}."""
    comments = {}
    with open(experimentPath) as fh:
        for line in fh:
            if not line.startswith("#"):
                continue
            body = line.lstrip("#").strip()
            if "=" in body:
                key, _, value = body.partition("=")
                comments[key.strip()] = value.strip()
    return comments


def resolveSurrogatePath(rawPath, qeccDataRoot):
    """A checkpoint path logged on a remote box may not exist verbatim locally.

    Try it as-is; otherwise re-root its '.../supervisedLearning/...' tail under qeccDataRoot.
    """
    if rawPath and os.path.isfile(rawPath):
        return rawPath
    if rawPath and "supervisedLearning" in rawPath:
        tail = rawPath[rawPath.index("supervisedLearning"):]
        candidate = os.path.join(qeccDataRoot, tail)
        if os.path.isfile(candidate):
            return candidate
    return rawPath  # let the caller fail loudly with the original path


def crawlRuns(qeccDataRoot):
    """Yield dicts describing every hybrid run folder under qeccDataRoot.

    A run folder has both policy_weights.pth and experiment.txt. Non-hybrid runs are
    yielded with skip='...'; the driver reports them but does not evaluate them.
    """
    for dirpath, _dirnames, filenames in os.walk(qeccDataRoot):
        if "policy_weights.pth" not in filenames or "experiment.txt" not in filenames:
            continue
        experimentPath = os.path.join(dirpath, "experiment.txt")
        comments = parseExperimentComments(experimentPath)
        useDict = comments.get("env_use_dict_observation", "False").lower() == "true"
        run = {
            "folder": dirpath,
            "name": os.path.basename(dirpath.rstrip("/\\")),
            "policyWeights": os.path.join(dirpath, "policy_weights.pth"),
            "comments": comments,
            "l": int(comments["env_l"]) if "env_l" in comments else None,
            "m": int(comments["env_m"]) if "env_m" in comments else None,
        }
        if not useDict:
            run["skip"] = "not a hybrid run (env_use_dict_observation != True)"
        yield run


def buildPolicyNet(run, qeccDataRoot, device="cpu"):
    """Reconstruct the deterministic hybrid actor and load its trained weights.

    The saved policy_weights.pth is policy_module.state_dict() with every tensor under the
    prefix 'module.0.module.<hybridNet param>' (verified against a real checkpoint). We strip
    that prefix and load into a bare hybridNet, so no torchrl scaffolding is needed at eval
    time. usePretrainedEncoderWeights=False because the trained policy weights overwrite the
    encoder anyway; the surrogate checkpoint is still needed for the architecture hyperparameters.
    """
    comments = run["comments"]
    l, m = run["l"], run["m"]
    kMin = int(comments.get("env_minimum_number_of_qubits", 1))
    numCells = int(comments.get("num_cells", 256))
    surrogatePath = resolveSurrogatePath(comments.get("model_surrogate_model_path"), qeccDataRoot)
    if not surrogatePath or not os.path.isfile(surrogatePath):
        raise FileNotFoundError(
            f"Surrogate checkpoint needed for architecture hyperparameters not found: "
            f"{comments.get('model_surrogate_model_path')!r} (also tried re-rooting under "
            f"{qeccDataRoot!r}).")
    outputSize = 2 * (l + 1) + 2 * (m + 1)

    net = hybridNet(l, m,
                    outputSize=outputSize,
                    minimumNumberOfQubits=kMin,
                    surrogateModelPath=surrogatePath,
                    num_cells=numCells,
                    device=device,
                    usePretrainedEncoderWeights=False)

    saved = torch.load(run["policyWeights"], map_location=device, weights_only=False)
    prefix = "module.0.module."
    hybridState = {key[len(prefix):]: value for key, value in saved.items()
                   if key.startswith(prefix)}
    if len(hybridState) != len(list(net.state_dict())):
        raise RuntimeError(
            f"policy weight key mismatch: stripped {len(hybridState)} of {len(saved)} saved "
            f"tensors, but the net has {len(list(net.state_dict()))}. Prefix assumption "
            f"'{prefix}' may be wrong for this checkpoint.")
    net.load_state_dict(hybridState, strict=True)
    net.eval()
    return net, kMin


def deterministicAction(net, obs, l, m):
    """Greedy action = mode of the block one-hot distribution over the policy logits."""
    with torch.no_grad():
        logits = net(
            torch.as_tensor(obs["aX"], dtype=torch.float32),
            torch.as_tensor(obs["bX"], dtype=torch.float32),
            torch.as_tensor(obs["aY"], dtype=torch.float32),
            torch.as_tensor(obs["bY"], dtype=torch.float32),
            torch.as_tensor(obs["code"], dtype=torch.float32),
            torch.as_tensor(obs["k"], dtype=torch.float32),
        )
        distribution = ConcatenatedOneHotCategorical(
            logits, blockSizes=[l + 1, l + 1, m + 1, m + 1])
        return distribution.mode.cpu().numpy().astype(np.int64)


def makeEnv(l, m, kMin, bitFlipping, numberOfSamples, numberOfIterations, errorRange):
    """A single environment instance driven directly (no torchrl), dict observation on."""
    return bicycleBivariateCodeEnvironment(
        l=l, m=m, errorRange=errorRange, minimumNumberOfLogicalQubits=kMin,
        numberOfSamples=numberOfSamples, numberOfIterations=numberOfIterations,
        rewardEngineering=False, codeLogging=False, bitFlipping=bitFlipping,
        useDictObservation=True)


def seedEnvWithCode(env, aX, bX, aY, bY):
    """Set the environment to a non-zero starting code (option (i): direct state set).

    reset() zeroes the polynomials and ignores options today; this helper overwrites the
    internal polynomial state and recomputes the derived matrices exactly as the env does.
    It collapses to a reset(options=...) call once the environment supports it.
    """
    env.aX = np.asarray(aX, dtype=env.aX.dtype)
    env.bX = np.asarray(bX, dtype=env.bX.dtype)
    env.aY = np.asarray(aY, dtype=env.aY.dtype)
    env.bY = np.asarray(bY, dtype=env.bY.dtype)
    aXi, aYi, bXi, bYi = polynomialsToIndices(env.aX, env.bX, env.aY, env.bY)
    env.A, env.B = generateABmatrices(env._l, env._m, aXi, aYi, bXi, bYi)
    env.Hx, env.Hz = bicycleCodeFromAB(env.A, env.B)
    env.numberOfLogicalQubits = calculateCodeDimension(env.Hx, env.Hz)
    return env._getObservation()


def rollout(net, env, l, m, maxSteps, startObs):
    """Deterministic rollout; record the visited (aX,bX,aY,bY) sequence, stop on first repeat.

    Returns (visitedCodes, cycleLength) where visitedCodes is a list of 0/1 coefficient
    tuples in (aX, bX, aY, bY) order and cycleLength is the step index of the first revisit
    (or maxSteps if none within budget).
    """
    obs = startObs
    visited = []
    seenStates = {}
    cycleLength = maxSteps
    for step in range(maxSteps + 1):
        state = (tuple(int(b) for b in env.aX), tuple(int(b) for b in env.bX),
                 tuple(int(b) for b in env.aY), tuple(int(b) for b in env.bY))
        if state in seenStates:
            cycleLength = step - seenStates[state]
            break
        seenStates[state] = step
        visited.append(state)
        if step == maxSteps:
            break
        action = deterministicAction(net, obs, l, m)
        obs, _reward, _terminated, _truncated, _info = env.step(action)
    return visited, cycleLength


def worstLoggedCodes(qeccDataRoot, l, m, kMin, numWorst):
    """The numWorst lowest-reward logged codes with k >= kMin, as 0/1 coefficient vectors.

    Bits arrive folded in [aX | aY | bX | bY] order (period l for aX/bX, m for aY/bY).
    """
    
    data = loadCodeEvaluations(dataRoot, l, m)
    rewards = rewardFromCounts(data.counts, data.samples, FIVE_POINT_GRID)
    eligible = np.where(data.k >= kMin)[0]
    if len(eligible) == 0:
        return []
    order = eligible[np.argsort(rewards[eligible])]      # ascending reward
    chosen = order[:numWorst]
    codes = []
    for index in chosen:
        bits = data.bits[index]
        aX = bits[0:l]
        aY = bits[l:l + m]
        bX = bits[l + m:2 * l + m]
        bY = bits[2 * l + m:2 * l + 2 * m]
        codes.append((np.array(aX), np.array(bX), np.array(aY), np.array(bY),
                      float(rewards[index])))
    return codes


# =====================================================================================
# Stage 2: decoders, failure-count sampling, structural metrics, distance
# =====================================================================================

def buildDecoders(decoderNames, maxIterations):
    """Return {name: (decoderFunction, dualBinary)} for the requested decoders.

    bposd_wN : dual-binary BP + ordered-statistics post-processing of order N.
    qbp      : quaternary refined belief propagation (single combined check matrix).
    """
    registry = {}
    for name in decoderNames:
        if name.startswith("bposd_w"):
            order = int(name[len("bposd_w"):])
            osdMethod = "osd0" if order == 0 else "osd_cs"
            binaryDecoder = functools.partial(
                wrapperForRoffesLdpc, osd_method=osdMethod, osd_order=order)
            registry[name] = (binaryDecoderToDualBinaryDecoderWrapper(binaryDecoder), True)
        elif name == "qbp":
            from qecc.qbp import refinedBPalgorithm3
            registry[name] = (refinedBPalgorithm3, False)
        else:
            raise ValueError(f"Unknown decoder {name!r}. Use bposd_w0..bposd_w5 or qbp.")
    return registry


def evaluateToFailureTarget(Hx, Hz, decoderFunction, dualBinary, errorRange, maxIterations,
                            targetFailures, sampleCap, batchSize, seed):
    """Accumulate samples per BER point until targetFailures failures or sampleCap samples.

    decoderEvaluator returns per-point COUNTS of logical errors and decoder failures (not
    rates). We sample each point independently (single-element errorRange) so a high-BER
    point does not force extra sampling on a low-BER point. Returns per-point arrays:
    logicalCounts, failureCounts, samples.
    """
    numberOfPoints = len(errorRange)
    logicalCounts = np.zeros(numberOfPoints)
    failureCounts = np.zeros(numberOfPoints)
    samples = np.zeros(numberOfPoints, dtype=np.int64)
    localSeed = seed
    for pointIndex, p in enumerate(errorRange):
        while True:
            logical, failure = decoderEvaluator(
                decoderFunction=decoderFunction, dualBinary=dualBinary,
                Hx=Hx, Hz=Hz, errorRange=[p], decoderStoppingCriterion=maxIterations,
                numberOfSamples=batchSize, seed=localSeed % (2**32 - 1))
            localSeed += 1
            logicalCounts[pointIndex] += logical[0]
            failureCounts[pointIndex] += failure[0]
            samples[pointIndex] += batchSize
            observedFailures = logicalCounts[pointIndex] + failureCounts[pointIndex]
            if observedFailures >= targetFailures or samples[pointIndex] >= sampleCap:
                break
    return logicalCounts, failureCounts, samples


def rewardFromEvaluation(logicalCounts, failureCounts, samples, errorRange):
    """Environment reward: trapezoid(1 - combinedBER) over the grid."""
    combined = (logicalCounts + failureCounts) / samples
    return float(np.trapezoid(1.0 - combined, np.asarray(errorRange, dtype=float)))


def bootstrapRewardCI(logicalCounts, failureCounts, samples, errorRange,
                      numberOfResamples=1000, seed=0):
    """95% bootstrap CI on the trapezoid reward, resampling the per-point failure counts.

    Each point's failures are Binomial(samples, p_hat); we resample p per point and
    recompute the reward, following the bootstrap-over-Monte-Carlo-uncertainty approach.
    """
    rng = np.random.default_rng(seed)
    combined = logicalCounts + failureCounts
    rewards = np.empty(numberOfResamples)
    grid = np.asarray(errorRange, dtype=float)
    for resampleIndex in range(numberOfResamples):
        resampledFailures = rng.binomial(samples, combined / samples)
        ber = resampledFailures / samples
        rewards[resampleIndex] = np.trapezoid(1.0 - ber, grid)
    return float(np.percentile(rewards, 2.5)), float(np.percentile(rewards, 97.5))


def structuralMetrics(l, m, aX, bX, aY, bY, Hx, Hz):
    """n, k, rate (and its inverse-integer rounding), check weight, syndrome depth."""
    n = Hx.shape[1]
    k = int(calculateCodeDimension(Hx, Hz))
    weight = checkWeight(aX, bX, aY, bY)
    rate, inverse = rateToInverseInteger(k, n)
    return dict(n=n, k=k, rate=rate, rateInverse=inverse,
                checkWeight=weight, syndromeDepth=weight + 1)


def codeDistanceMIP(Hx, Hz, timeBudgetSeconds):
    """Exact minimum distance by mixed-integer programming (Landahl-Anderson-Rice / Bravyi).

    For each logical operator we minimise the Hamming weight of a representative subject to
    (a) commuting with the opposite-type stabilisers (in the kernel, linearised mod 2 with
    integer slacks) and (b) anticommuting with that logical (odd overlap). d = min(dX, dZ).
    Returns (lowerBound, upperBound, certified): on timeout the incumbent is the upper bound
    and the solver's dual bound the lower bound; certified is True when they meet.
    """
    from scipy.optimize import milp, LinearConstraint, Bounds

    logicalX, logicalZ = computeLogicals(Hx, Hz)

    def minimumWeightLogical(parityMatrix, logicalFunctionals, deadline):
        """min |v| : parityMatrix v = 0 (mod 2), functional . v = 1 (mod 2), over all functionals."""
        n = parityMatrix.shape[1]
        best = (0, np.inf, False)  # (lower, upper, certified) tracked as global-ish below
        globalLower, globalUpper, allCertified = 0, np.inf, True
        for functional in logicalFunctionals:
            remaining = deadline - time.time()
            if remaining <= 0:
                allCertified = False
                break
            rows = parityMatrix.shape[0]
            # variables: v (n binary), t (rows integer >=0 for parity slack), s (1 integer >=0)
            numVars = n + rows + 1
            objective = np.zeros(numVars)
            objective[:n] = 1.0
            # parity constraints: parityMatrix v - 2 t = 0
            parityBlock = np.hstack([parityMatrix,
                                     -2.0 * np.eye(rows),
                                     np.zeros((rows, 1))])
            parityConstraint = LinearConstraint(parityBlock, 0, 0)
            # odd-overlap: functional . v - 2 s = 1
            overlapRow = np.hstack([np.asarray(functional, dtype=float),
                                    np.zeros(rows), [-2.0]])
            overlapConstraint = LinearConstraint(overlapRow.reshape(1, -1), 1, 1)
            integrality = np.ones(numVars)  # all integer; v is 0/1 via bounds
            lower = np.zeros(numVars)
            upper = np.concatenate([np.ones(n), np.full(rows + 1, np.inf)])
            result = milp(c=objective, constraints=[parityConstraint, overlapConstraint],
                          integrality=integrality, bounds=Bounds(lower, upper),
                          options={"time_limit": max(0.1, remaining)})
            if result.success and result.x is not None:
                weight = int(round(result.x[:n].sum()))
                globalUpper = min(globalUpper, weight)
                if result.status == 0:                 # optimal, certified
                    globalLower = max(globalLower, 1)  # a nontrivial logical exists
                else:
                    allCertified = False
            else:
                allCertified = False
        return globalLower, globalUpper, allCertified

    deadline = time.time() + timeBudgetSeconds
    lowerZ, upperZ, certifiedZ = minimumWeightLogical(Hx, logicalX, deadline)  # dZ
    lowerX, upperX, certifiedX = minimumWeightLogical(Hz, logicalZ, deadline)  # dX
    upper = min(upperX, upperZ)
    certified = certifiedX and certifiedZ and np.isfinite(upper)
    lower = upper if certified else 1
    return lower, (int(upper) if np.isfinite(upper) else None), certified


# =====================================================================================
# Equivalence classing (item: are the best codes actually different?)
# =====================================================================================

def sameCode(codeA, codeB):
    """Two CSS codes are equal iff their X-checks span the same row space and likewise Z.

    rank([H1; H2]) == rank(H1) == rank(H2) for both the X and Z parity matrices (GF(2)).
    """
    HxA, HzA = codeA
    HxB, HzB = codeB
    if HxA.shape[1] != HxB.shape[1]:
        return False
    for H1, H2 in ((HxA, HxB), (HzA, HzB)):
        rank1, rank2 = gf2Rank(H1), gf2Rank(H2)
        rankStacked = gf2Rank(np.vstack([H1, H2]))
        if not (rankStacked == rank1 == rank2):
            return False
    return True


def equivalenceClasses(codesWithMeta):
    """Group codes into distinct-code classes; cheap signature bucketing before the rank test.

    codesWithMeta: list of dicts each with 'Hx', 'Hz', 'signature' (n,k,checkWeight).
    Returns a list of class indices, one per input code.
    """
    classIds = [-1] * len(codesWithMeta)
    representatives = []  # (signature, (Hx, Hz), classId)
    nextClass = 0
    for index, item in enumerate(codesWithMeta):
        signature = item["signature"]
        assigned = None
        for repSignature, repCode, repClass in representatives:
            if repSignature != signature:
                continue                       # different signature => trivially distinct
            if sameCode((item["Hx"], item["Hz"]), repCode):
                assigned = repClass
                break
        if assigned is None:
            assigned = nextClass
            representatives.append((signature, (item["Hx"], item["Hz"]), assigned))
            nextClass += 1
        classIds[index] = assigned
    return classIds


# =====================================================================================
# Driver
# =====================================================================================

@dataclass
class VisitedCode:
    l: int
    m: int
    aX: np.ndarray
    bX: np.ndarray
    aY: np.ndarray
    bY: np.ndarray
    origin: str                 # "zero" or "worst-<i>"
    firstStep: int              # step index at which this code was first visited
    runName: str
    signature: tuple = None
    Hx: np.ndarray = None
    Hz: np.ndarray = None
    results: dict = field(default_factory=dict)   # (decoder, gridName) -> metrics


def codeKey(vc):
    """Dedup key: the folded coefficient tuple in (aX,bX,aY,bY) order plus size."""
    return (vc.l, vc.m, tuple(int(b) for b in vc.aX), tuple(int(b) for b in vc.bX),
            tuple(int(b) for b in vc.aY), tuple(int(b) for b in vc.bY))


def runStage1(run, qeccDataRoot, rolloutMultiplier, numWorst, numberOfIterations):
    """Load the policy and produce the list of visited codes (zero start + worst starts)."""
    l, m = run["l"], run["m"]
    bitFlipping = run["comments"].get("env_bit_flipping", "False").lower() == "true"
    net, kMin = buildPolicyNet(run, qeccDataRoot)
    maxSteps = rolloutMultiplier * max(l, m)
    visited = []

    env = makeEnv(l, m, kMin, bitFlipping, numberOfSamples=1,
                  numberOfIterations=numberOfIterations, errorRange=FIVE_POINT_GRID)
    obs, _ = env.reset(seed=0)
    zeroVisited, zeroCycle = rollout(net, env, l, m, maxSteps, obs)
    for step, state in enumerate(zeroVisited):
        aX, bX, aY, bY = state
        visited.append(VisitedCode(l, m, np.array(aX), np.array(bX), np.array(aY),
                                   np.array(bY), "zero", step, run["name"]))

    try:
        worst = worstLoggedCodes(qeccDataRoot, l, m, kMin, numWorst) if numWorst > 0 else []
    except (ValueError, FileNotFoundError) as error:
        print(f"  item (d) skipped for {run['name']}: no usable logged codes ({error})")
        worst = []
    for worstIndex, (aX, bX, aY, bY, _reward) in enumerate(worst):
        env.reset(seed=worstIndex + 1)
        startObs = seedEnvWithCode(env, aX, bX, aY, bY)
        worstVisited, _cycle = rollout(net, env, l, m, maxSteps, startObs)
        for step, state in enumerate(worstVisited):
            vaX, vbX, vaY, vbY = state
            visited.append(VisitedCode(l, m, np.array(vaX), np.array(vbX), np.array(vaY),
                                       np.array(vbY), f"worst-{worstIndex}", step, run["name"]))
    return visited, zeroCycle


def distinctCodes(visited):
    """Deduplicate visited codes, keeping the earliest (origin, firstStep) as the record."""
    byKey = {}
    for vc in visited:
        key = codeKey(vc)
        if key not in byKey or vc.firstStep < byKey[key].firstStep:
            byKey[key] = vc
    return list(byKey.values())


def evaluateDistinctCode(vc, decoders, grids, targetFailures, sampleCap, batchSize,
                         numberOfIterations, seed):
    """Fill vc.Hx/Hz/signature and vc.results for every (decoder, grid)."""
    vc.Hx, vc.Hz = buildCode(vc.l, vc.m, vc.aX, vc.bX, vc.aY, vc.bY)
    structural = structuralMetrics(vc.l, vc.m, vc.aX, vc.bX, vc.aY, vc.bY, vc.Hx, vc.Hz)
    vc.signature = (structural["n"], structural["k"], structural["checkWeight"])
    vc.structural = structural
    if structural["k"] == 0:
        # No logical operators: the decoder sweep is undefined (and computeLogicals is empty).
        # Keep the structural row; leave vc.results empty. Mirrors the env, which only decodes
        # a code that has logical qubits.
        return vc
    for decoderName, (decoderFunction, dualBinary) in decoders.items():
        for gridName, grid in grids.items():
            logical, failure, samples = evaluateToFailureTarget(
                vc.Hx, vc.Hz, decoderFunction, dualBinary, grid, numberOfIterations,
                targetFailures, sampleCap, batchSize, seed)
            reward = rewardFromEvaluation(logical, failure, samples, grid)
            ciLow, ciHigh = bootstrapRewardCI(logical, failure, samples, grid)
            vc.results[(decoderName, gridName)] = dict(
                reward=reward, ciLow=ciLow, ciHigh=ciHigh,
                logicalCounts=logical, failureCounts=failure, samples=samples)
    return vc


def referenceVisitedCodes(l, m):
    """The published reference code for (l, m) as a VisitedCode, if we have one."""
    spec = REFERENCE_CODES.get((l, m))
    if spec is None:
        return []
    aX = np.zeros(l, dtype=np.int64); aX[spec["aX"]] = 1
    bX = np.zeros(l, dtype=np.int64); bX[spec["bX"]] = 1
    aY = np.zeros(m, dtype=np.int64); aY[spec["aY"]] = 1
    bY = np.zeros(m, dtype=np.int64); bY[spec["bY"]] = 1
    return [VisitedCode(l, m, aX, bX, aY, bY, f"reference:{spec['name']}", -1, "reference")]


REPORT_KEYS = [
    "run", "origin", "firstStep", "decoder", "grid",
    "n", "k", "rate", "checkWeight", "syndromeDepth",
    "reward", "rewardCiLow", "rewardCiHigh",
    "distinctClass", "distanceLower", "distanceUpper", "distanceCertified",
]


def writeReport(outputPath, records):
    """Reuse the project logger for a tab-separated report readable by postProcessingRL tooling."""
    directory = os.path.dirname(outputPath) or "."
    reportLogger = logger(keys=REPORT_KEYS, logPath=directory,
                          fileName=os.path.basename(outputPath))
    for record in records:
        for key in REPORT_KEYS:
            reportLogger.keyValue(key, record.get(key, ""))
        reportLogger.dumpLogger(printOut=False)
    return reportLogger.fullPath


def buildArgumentParser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qecc-data-root", default=os.environ.get("QECC_DATA"),
                        help="Root under which run folders and supervisedLearning/ live. "
                             "Defaults to $QECC_DATA.")
    parser.add_argument("--output", default=None,
                        help="Report path. Default: policyEvaluation.txt inside each run folder.")
    parser.add_argument("--rollout-multiplier", type=int, default=10,
                        help="Rollout length is this times max(l, m). Prefixes give the "
                             "k*max(l,m) sweep for k=0..multiplier.")
    parser.add_argument("--num-worst", type=int, default=10,
                        help="Number of worst logged codes to also roll out from (item d).")
    parser.add_argument("--target-failures", type=int, default=100,
                        help="Sample each BER point until this many logical failures observed.")
    parser.add_argument("--sample-cap", type=int, default=200000,
                        help="Maximum samples per BER point (protects low-BER points).")
    parser.add_argument("--batch-size", type=int, default=200,
                        help="Samples per decoderEvaluator call while accumulating failures.")
    parser.add_argument("--grids", default="5,10", choices=["5", "10", "5,10", "10,5"],
                        help="Which noise grids to evaluate on.")
    parser.add_argument("--decoders",
                        default="bposd_w0,bposd_w1,bposd_w2,bposd_w3,bposd_w4,bposd_w5,qbp",
                        help="Comma-separated decoder names.")
    parser.add_argument("--num-decoder-iterations", type=int, default=50,
                        help="Max BP iterations for every decoder.")
    parser.add_argument("--distance-topn", type=int, default=0,
                        help="Compute exact MIP distance for the top-N codes per run (0 = off).")
    parser.add_argument("--distance-time-budget", type=float, default=600.0,
                        help="Per-code wall-clock budget (seconds) for the distance MIP.")
    parser.add_argument("--num-processes", type=int, default=1,
                        help="Worker processes for Stage 2 over the distinct-code list.")
    return parser


def selectGrids(gridsArgument):
    mapping = {"5": ("grid5", FIVE_POINT_GRID), "10": ("grid10", TEN_POINT_GRID)}
    return dict(mapping[token] for token in gridsArgument.split(","))


def main():
    args = buildArgumentParser().parse_args()
    if not args.qecc_data_root:
        raise SystemExit("No data root: pass --qecc-data-root or set $QECC_DATA.")
    grids = selectGrids(args.grids)
    decoderNames = [name.strip() for name in args.decoders.split(",") if name.strip()]
    decoders = buildDecoders(decoderNames, args.num_decoder_iterations)

    for run in crawlRuns(args.qecc_data_root):
        if run.get("skip"):
            print(f"SKIP {run['name']}: {run['skip']}")
            continue
        print(f"=== run {run['name']}  (l={run['l']}, m={run['m']}) ===")
        visited, zeroCycle = runStage1(
            run, args.qecc_data_root, args.rollout_multiplier, args.num_worst,
            args.num_decoder_iterations)
        pool = distinctCodes(visited) + referenceVisitedCodes(run["l"], run["m"])
        print(f"  {len(visited)} visited, {len(pool)} distinct (incl. reference); "
              f"zero-start cycle length {zeroCycle}")

        evaluate = functools.partial(
            evaluateDistinctCode, decoders=decoders, grids=grids,
            targetFailures=args.target_failures, sampleCap=args.sample_cap,
            batchSize=args.batch_size, numberOfIterations=args.num_decoder_iterations, seed=0)
        if args.num_processes > 1:
            import multiprocessing
            with multiprocessing.Pool(args.num_processes) as workerPool:
                pool = workerPool.map(evaluate, pool)
        else:
            pool = [evaluate(vc) for vc in pool]

        classIds = equivalenceClasses(
            [dict(Hx=vc.Hx, Hz=vc.Hz, signature=vc.signature) for vc in pool])

        # optional exact distance for the top-N by best reward across decoders/grids
        distanceByCode = {}
        if args.distance_topn > 0:
            def bestReward(vc):
                return max((r["reward"] for r in vc.results.values()), default=-np.inf)
            ranked = sorted(range(len(pool)), key=lambda i: bestReward(pool[i]), reverse=True)
            for i in ranked[:args.distance_topn]:
                distanceByCode[i] = codeDistanceMIP(pool[i].Hx, pool[i].Hz,
                                                    args.distance_time_budget)

        records = []
        for codeIndex, vc in enumerate(pool):
            lower, upper, certified = distanceByCode.get(codeIndex, ("", "", ""))
            structuralRow = dict(
                run=vc.runName, origin=vc.origin, firstStep=vc.firstStep,
                n=vc.structural["n"], k=vc.structural["k"],
                rate=round(vc.structural["rate"], 6), checkWeight=vc.structural["checkWeight"],
                syndromeDepth=vc.structural["syndromeDepth"],
                distinctClass=classIds[codeIndex],
                distanceLower=lower, distanceUpper=upper, distanceCertified=certified)
            if not vc.results:
                # k == 0: no decoder sweep. Emit one structural-only row so the code still appears.
                records.append(dict(structuralRow, decoder="(none: k=0)", grid=""))
                continue
            for (decoderName, gridName), metrics in vc.results.items():
                records.append(dict(
                    structuralRow, decoder=decoderName, grid=gridName,
                    reward=round(metrics["reward"], 6),
                    rewardCiLow=round(metrics["ciLow"], 6),
                    rewardCiHigh=round(metrics["ciHigh"], 6)))

        outputPath = args.output or os.path.join(run["folder"], "policyEvaluation.txt")
        written = writeReport(outputPath, records)
        print(f"  wrote {len(records)} rows to {written}")


if __name__ == "__main__":
    main()
