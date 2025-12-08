"""
Fun with matrices, in honour of Dr. James Wotton's brilliant video lecture
"""

from sympy.matrices import Matrix, eye, zeros, ones, diag, GramSchmidt
from sympy import symbols, shape, init_printing, Basic, And, Xor, Not, simplify_logic
import numpy as np
import matplotlib.pyplot as plt
#from qecc.polynomialCodes import A1_HX, A1_HZ, A2_HX, A2_HZ, A3_HX, A3_HZ, A4_HX, A4_HZ, A5_HX, A5_HZ,  A6_HX, A6_HZ
seed = 777
localRandom = np.random.RandomState(seed)
from numpy.linalg import matrix_rank
import copy

def generateSymbolicMatrix(rows, cols):
    symbolsList = []
    symbolicRows = []

    for i in range(rows):
        newRow = []
        for j in range(cols):
            stringLiteral = f"a{i}{j}"
            s = symbols(stringLiteral)
            symbolsList.append(s) 
            newRow.append(s)
        symbolicRows.append(newRow)
    print(symbolsList)
    matrix = Matrix(rows, cols, symbolsList)
    return matrix, symbolsList, symbolicRows

def symbolicDeterminant(matrix):
    # Calculate the determinant of a symbolic matrix over F(2) using the recursive definition: \xor (-1)^{1+j} matrix_{1j} det(M_{1j})
    if matrix.shape[0] == 2:
        return Xor(And(matrix[0,0],matrix[1,1]), And(matrix[0,1], matrix[1,0]))
    else:
        expression = And(matrix[0,0], symbolicDeterminant(matrix[1:,1:]))
        for j in range(1, matrix.shape[1]):
            #print(Matrix.hstack(matrix[1:,:j], matrix[1:,j+1:]))
            expression = Xor(expression, And(matrix[0,j] , symbolicDeterminant(Matrix.hstack(matrix[1:,:j],matrix[1:,j+1:]))))
    return expression

def symbolicCofactor(matrix):
    # Calculate the cofactor matrix of a symbolic matrix over F(2)
    #cofactorsList = [f"c{i}{j}" for i in range( matrix.shape[0]) for j in range(matrix.shape[1])]
    #cofactorMatrix = Matrix(matrix.shape[0], matrix.shape[1], cofactorsList)
   
    cofactorsList = []
    for i in range(matrix.shape[0]):
        newRow = []
        for j in range(matrix.shape[1]):
            minor = matrix.copy() # Not very efficient, but good enough for one time use.
            minor.col_del(j)
            minor.row_del(i)
            newRow.append(symbolicDeterminant(minor))
        cofactorsList.append(newRow)
    return cofactorsList


def binaryGaussianEliminationOnRows(matrix, returnDtype = np.int32):
    matrixInverse = np.eye(matrix.shape[0], dtype = returnDtype)
    rank = 0
    for k in range(0, matrix.shape[0], 1): #OMer: changed from matrix.shape[1] to matrix.shape[0]
        # Find the first row index i >= k, such that row i has non zero element at column k
        for i in range(k, matrix.shape[0], 1):
            if matrix[i, k] != 0:
                temp = copy.copy(matrix[rank, :])
                matrix[rank, :] = matrix[i, :]
                matrix[i, :] = copy.copy(temp)
                
                temp = copy.copy(matrixInverse[rank,:])
                matrixInverse[rank,:] = matrixInverse[i,:]
                matrixInverse[i, :] = copy.copy(temp)
                
                # Now eliminate any rows, j>i,  below row i, that have matrix[j,k] != 0
                for j in range(0, matrix.shape[0], 1):
                    if j != rank:
                        if matrix[j, k] != 0:
                            
                            matrix[j, :] = (matrix[j, :] + matrix[rank, :]) %2
                            matrixInverse[j, :] = (matrixInverse[j, :] + matrixInverse[rank, :]) %2#Omer: changhed from matrix to matrixInverse
                rank += 1
    return matrix, matrixInverse, rank

def binaryDeterminant(matrix, rowNumber = 0):

    
    if not (matrix.shape[0] == matrix.shape[1]):
        raise("Determinant of matrices is only supported for squrae matrices.")
    
    if matrix.shape[0] == 2:
        determinantResult =  (matrix[0,0] * matrix[1,1]) + (matrix[0,1] * matrix[1,0]) %2
    else:
        # Determinant according to row number 0
        determinantResult = False
        
        for col in range(matrix.shape[1]):
            cofactorMatrix = np.hstack( (np.vstack((matrix[:rowNumber, :col], matrix[rowNumber+1: ,:col])) , np.vstack( (matrix[:rowNumber, col+1:], matrix[rowNumber+1:, col+1:]))))
            determinantResult ^= ( matrix[rowNumber, col] & binaryDeterminant(cofactorMatrix))  # &(-1) ** (i+j) which equates to 1 over F(2)
    return determinantResult
    

if __name__ == "__main__":
    pass

    