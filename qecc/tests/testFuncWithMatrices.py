import src.funWithMatrices as funWithMatrices
import numpy as np
from src.polynomialCodes import A1_HX
import copy

def test_identityMatrix(size = 5):
    matrix = np.eye(size, dtype = np.bool)
    assert (funWithMatrices.binaryDeterminant(matrix) == True)
    matrix, matrixInverse, rank = funWithMatrices.binaryGaussianEliminationOnRows(matrix)
    assert(np.all(matrix == matrixInverse))
    assert(np.all(matrix == np.eye(5)))

def test_permutationMatrix(size = 5):
    identityMatrix = np.eye(size, dtype=np.bool)

    from itertools import permutations
    for p in permutations(range(size)):
    #p = list(permutations(range(size)))[1]
    #print(f"Permutation p == {p}")
        indexPermutation = [p[i] for i in range(size)]
        permutationMatrix = identityMatrix[indexPermutation, : ]
        assert(funWithMatrices.binaryDeterminant(permutationMatrix) == True)
        matrix, matrixInverse, rank = funWithMatrices.binaryGaussianEliminationOnRows(permutationMatrix)
        assert(np.all(matrix == np.eye(size, dtype = np.bool)))
        assert(np.all(identityMatrix[indexPermutation, : ] @ matrixInverse == np.eye(size, dtype = np.bool)))


def test_A1_HX():
    import time
    parity = copy.copy(A1_HX[:,0:127]).astype(np.bool)
    

    for i in range(64):
        start = time.time()
        bd = funWithMatrices.binaryDeterminant(parity[0:i,0:i])
        print(bd)
        end = time.time()
        print(f"Time it too to calc det of matrix of size {i}X{i} is {end-start}.")

        # Create a stream to capture the profiling results
    matrix, matrixInverse, rank = funWithMatrices.binaryGaussianEliminationOnRows(parity)
    assert(rank == 113)
    assert(np.all(matrix[0:rank,0:rank] == np.eye(rank, dtype = np.bool)))
    

def test_hammingParityMatrix():
    H = np.array([[1,1,1,0,0,1,1,0,0,1],
                [1,0,1,0,1,1,0,1,1,0],
                [0,0,1,1,1,0,1,0,1,1],
                [0,1,0,1,1,1,0,1,0,1], 
                [1,1,0,1,0,0,1,1,1,0]])

    matrix, matrixInverse, rank = funWithMatrices.binaryGaussianEliminationOnRows(H.astype(np.bool))
    for i in range(5):
        assert( matrix[i,i] == True)
    assert(rank == 5)



if __name__ == "__main__":
    test_identityMatrix()
    test_permutationMatrix(5)
    test_hammingParityMatrix()
    test_A1_HX()
    

   