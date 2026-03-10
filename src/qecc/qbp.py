from qecc.gf4 import *

from qecc.polynomialCodes import codes
from itertools import permutations, product
from numba import jit
from line_profiler import profile
from numba import jit
from numba.experimental import jitclass
from numba.types import int32, float64, DictType


#@jit(nopython=True)
def generateSolutionsArray(array):
    """
    array is assumed to be a one dimensional array over the set {0,1,2,3}
    """
    gf4Ints = [0,1,2,3]
    GF4 = [gf4(i) for i in gf4Ints]
    solutions = {0: [], 1: []}
    #if not isinstance(array,np.ndarray):
    #    raise ValueError("array must be an ndarray")
    nonZeros = np.where(array != 0)[0]
    arrayAsGf4 = list(map(gf4, array[nonZeros]))
    candidates = gf4CartesianProduct(len(nonZeros))
    for c in candidates:
        result = traceInner(arrayAsGf4,c)
        solutions[result.value].append(toarray(c))
    return solutions

#@jit(nopython=True)
def generateSolutions(H):
    """
    H is assumed to be a two dimensional array over the set {0,1,2,3}
    """
    solutions = {}
    #if not isinstance(H,np.ndarray):
    #    raise ValueError("H must be an ndarray")
    for row in range(H.shape[0]):
        print(f"Generating solutions for row {row} of H")
        solutions[row] = generateSolutionsArray(H[row,:])
    return solutions

checkNodeSpec = [
    ('rowNumber', int32),
    ('hRow', int32[:]),
    ('nonZero_indices', int32[:]),
    ('solutions', DictType(int32, int32[:])),
]

@jitclass(checkNodeSpec)
class checkNode:
    def __init__(self, rowNumber, hRow):
        self.rowNumber = rowNumber
        self.hRow = hRow
        self.nonZero_indices = np.where(hRow != 0)[0]
        self.solutions = generateSolutionsArray(hRow)
    
    def step(self, probabilities):
        """
        Input:
        probabilities: array of shape (num_connected_variables, 4) with probabilities for each Pauli
        Output:
        r: messages to send back to variable nodes
        """
        num_vars = probabilities.shape[0]
        r = np.zeros((num_vars, 4))
        
        for n in range(num_vars):
            for pauli_idx in range(4):
                solution_indices = self.solutions[pauli_idx]
                for idx in solution_indices:
                    r[n][pauli_idx] += np.prod(probabilities[idx])
        
        return r


def refinedBPalgorithm3(H, s, initialValues, maxNumberOfIteration = 10):
    """
    Straight forward implementation of algorithm 3 from "Refined Belief Propagation Decoding of
    Sparse-Graph Quantum Codes"
    Arguments:
    H: (int) M X N parity check matrix with entries in {0,1,2,3} (no safety checks are performed). It is assumed that H is a matrix for a CSS code, i.e., that it has the form [[H_x, 0], [0,2*H_z]] where H_x and H_z are binary (Omer: need to check this). S in the paper is H here.
    s:  (int) syndrome vector of length M with entries in {0,1} where s[i] = 0 if the error commutes with the i-th stabilizer (check) and s[i] = 1 if the error anti-commutes with the i-th stabilizer. z in the paper is s here.
    initialValues: (float) initial values array of size 4XN. p in the paper is initialValues here
    maxNumberOfIteration: (int) maximum number of iterations for the algorithm

    returns:
    E: (int) array of length N with entries in {0,1,2,3} representing the most likely error according to the algorithm
    success: (bool) whether the algorithm succeeded in finding an error consistent with the syndrome
    """
    # It could be we would like to use an optimized choice of float, so I'm placing its type here as a constant.
    FLOAT_TYPE = np.float64
    BINARY_TYPE = np.int32
    INTEGER_TYPE = np.int32
    #I = 0
    #X = 1
    #Y = 2
    #Z = 3
    M,N = H.shape
    # There is a massive overloading of the letter q in the paper, so I'm replacing the last (two dimension) q with Q
    q = np.zeros((H.shape[0], H.shape[1], 4), dtype = FLOAT_TYPE)
    Q = np.zeros((H.shape[1], 4), dtype = FLOAT_TYPE)
    q0 = np.zeros(H.shape, dtype = FLOAT_TYPE)
    q1 = np.zeros(H.shape, dtype = FLOAT_TYPE)
    r = np.zeros((2, H.shape[0], H.shape[1]), dtype = FLOAT_TYPE)
    d = np.zeros(H.shape, dtype = FLOAT_TYPE)
    E = np.zeros(H.shape[1], dtype = INTEGER_TYPE)
    tip = np.zeros(M, dtype = BINARY_TYPE)
    success = False

    
    #cnn is used for m
    #vnn is used for n
    iterationNumber = 0
    # Initialization:
    for vnn in range(N):
        for cnn in np.where(H[:,vnn] != 0)[0]:
            for w in [0,1,2,3]:
                q[cnn,vnn, w] = initialValues[vnn,w]
            q0[ cnn, vnn] = q[cnn, vnn, 0] + q[cnn, vnn, H[cnn, vnn]]
            q1[ cnn, vnn] = 1 - q0[cnn, vnn]
            d[cnn, vnn] = q0[cnn, vnn] - q1[cnn, vnn]
        
    while iterationNumber < maxNumberOfIteration and not success:
        iterationNumber += 1
        
        # Horizontal step:
        delta = np.zeros(H.shape, dtype = BINARY_TYPE)
        for cnn in range(M):
            variableNodeIndices = np.where(H[cnn,:] != 0)[0]
            # The following is a calculation in (+1)s and (-1)s binary:
            master = ((-1) ** (s[cnn])) * np.prod(d[cnn,variableNodeIndices])
            for vnn in variableNodeIndices:
                delta[cnn, vnn] = master / d[cnn, vnn]
        
        # Vertical step:
        for vnn in range(N):
            checkNodeIndices = np.where(H[:,vnn] !=0 )[0]
            for cnn in checkNodeIndices:
                r[0,cnn,vnn] = (1 + delta[cnn, vnn]) / 2
                r[1,cnn,vnn] = (1 - delta[cnn, vnn]) / 2
                q[cnn, vnn,0] = initialValues[vnn][0] * np.prod([r[0,othercnn, vnn] for othercnn in checkNodeIndices if othercnn != cnn])
                for w in [1,2,3]:
                    q[cnn, vnn,w] = initialValues[vnn][w] * np.prod([ r[ integerTraceProduct[w][H[othercnn, vnn]] ,othercnn, vnn] for othercnn in checkNodeIndices if othercnn != cnn])
                #print(f"q0[cnn,vnn]: {q0[cnn,vnn]}")
                #print(f"q1[cnn,vnn]: {q1[cnn,vnn]}")
                a = 1/(q0[cnn,vnn] + q1[cnn,vnn])
                #print(f"a: {a}")
                q0[cnn, vnn] = a * (q[cnn, vnn, 0] + q[cnn,vnn,H[cnn, vnn]])
                q1[cnn, vnn] = a * np.sum([q[cnn, vnn, otherW] for otherW in [1,2,3] if otherW != H[cnn, vnn]])
        
        # Hard decision
        for vnn in range(N):
            checkNodeIndices = np.where(H[:,vnn] != 0)[0]
            Q[vnn,0] = initialValues[vnn,0] * np.prod(r[0, checkNodeIndices, vnn])
            for w in [1,2,3]:
                Q[vnn,w] = initialValues[vnn,w] * np.prod(r[integerTraceProduct[w][H[w, vnn]], checkNodeIndices, vnn])
        
            E[vnn] = np.argmax(Q[vnn,:])
        for m in range(M):
            tip[m] = np.sum([integerTraceProduct[E[n]][H[m, n]] for n in range(N)]) % 2
        if np.all(tip == s):
            success = True
    
    return E, success
    
    




if __name__ == "__main__":
    hX = codes["A1_HX"]
    H = np.array(hX)
    print(H[0:1,:])
    solutions = generateSolutions(H[0:1,:])

    