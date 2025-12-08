from qecc import funWithMatrices
import numpy as np
from qecc.polynomialCodes import A1_HX
import copy

def test_identityMatrix(size = 5):
    matrix = np.eye(size, dtype = np.int32)
    assert (funWithMatrices.binaryDeterminant(matrix) == True)
    matrix, matrixInverse, rank = funWithMatrices.binaryGaussianEliminationOnRows(matrix)
    assert(np.all(matrix == matrixInverse))
    assert(np.all(matrix == np.eye(5)))

def test_permutationMatrix(size = 5):
    identityMatrix = np.eye(size, dtype=np.int32)

    from itertools import permutations
    for p in permutations(range(size)):
        indexPermutation = [p[i] for i in range(size)]
        permutationMatrix = identityMatrix[indexPermutation, : ]
        assert(funWithMatrices.binaryDeterminant(permutationMatrix) == True)
        matrix, matrixInverse, rank = funWithMatrices.binaryGaussianEliminationOnRows(permutationMatrix)
        assert(np.all(matrix == np.eye(size, dtype = np.bool)))
        assert(np.all(identityMatrix[indexPermutation, : ] @ matrixInverse == np.eye(size, dtype = np.int32)))


def test_A1_HX():
    import time
    parity = copy.copy(A1_HX[:,0:127]).astype(np.int32)
    
    matrix, matrixInverse, rank = funWithMatrices.binaryGaussianEliminationOnRows(copy.copy(parity).astype(np.int32))

    assert(rank == 113)
    assert(np.all(matrix[0:rank,0:rank] == np.eye(rank, dtype = np.int32)))
    #assert(np.all(parity[0:rank, 0:rank] @ matrixInverse[0:rank,0:rank] %2 == np.eye(rank, dtype = np.bool)))

def test_hammingParityMatrix():
    H = np.array([[1,1,1,0,0,1,1,0,0,1],
                [1,0,1,0,1,1,0,1,1,0],
                [0,0,1,1,1,0,1,0,1,1],
                [0,1,0,1,1,1,0,1,0,1], 
                [1,1,0,1,0,0,1,1,1,0]])

    matrix, matrixInverse, rank = funWithMatrices.binaryGaussianEliminationOnRows(H.astype(np.int32))
    for i in range(5):
        assert( matrix[i,i] == True)
    assert(rank == 5)

def test_lastRowCancel():
    testMatrix = np.eye(3, dtype = np.int32)
    testMatrix[2,0] = 1
    matrix, matrixInverse, rank = funWithMatrices.binaryGaussianEliminationOnRows(copy.copy(testMatrix))
    assert(np.all(matrixInverse.dot(testMatrix) %2 == np.eye(3, dtype = np.int32)))

def test_solveBinaryLinearSystem():
    A = np.array([[1,0,0,0],
                  [0,1,0,0],
                  [0,0,1,0], 
                  [0,0,0,1]], dtype = np.int32)
    solutions = []
    solutions.append(np.zeros(4, dtype = np.int32))
    solutions = np.array(solutions)
    prposedSolutions = funWithMatrices.solveHomogenicBinaryLinearSystem(A)

    assert(np.all( prposedSolutions == solutions))
    A = np.array([[1,0,0,0],
                  [0,1,0,0],
                  [0,0,1,0]], dtype = np.int32)
    solutions = []
    x = np.zeros(4, dtype = np.int32)
    x[3] = 1
    solutions.append(x)
    solutions = np.array(solutions)
    proposedSolutions = funWithMatrices.solveHomogenicBinaryLinearSystem(A)
    assert(np.all( proposedSolutions == solutions))
    A = np.array([[1,0,0,0],
                  [0,1,0,0],
                  [0,0,1,1]], dtype = np.int32)
    solutions = []
    solution1 = np.zeros(4, dtype = np.int32)
    solution1[3] = 1
    solution1[2] = 1
    solutions.append(solution1)
    solutions = np.array(solutions)
    assert(np.all(funWithMatrices.solveHomogenicBinaryLinearSystem(A) == solutions))
    
    
if __name__ == "__main__":
    test_solveBinaryLinearSystem()
    test_identityMatrix()
    test_permutationMatrix(5)
    test_lastRowCancel()
    test_hammingParityMatrix()
    test_A1_HX()
 

    

   