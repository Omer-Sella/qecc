import funWithMatrices
import numpy as np

def test_identityMatrix(size = 5):
    matrix = np.eye(size, dtype = np.bool)
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

        matrix, matrixInverse, rank = funWithMatrices.binaryGaussianEliminationOnRows(permutationMatrix)
#        print(matrix)
 #       print(matrixInverse)
        assert(np.all(matrix == np.eye(size, dtype = np.bool)))
        assert(np.all(identityMatrix[indexPermutation, : ] @ matrixInverse == np.eye(size, dtype = np.bool)))


if __name__ == "__main__":

    test_permutationMatrix(5)
    test_identityMatrix()