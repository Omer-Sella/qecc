"""
Fun with matrices, in honour of Dr. James Wotton's brilliant video lecture
"""

from sympy.matrices import Matrix, eye, zeros, ones, diag, GramSchmidt
from sympy import symbols, shape, init_printing, Basic, And, Xor, Not, simplify_logic
import numpy as np
import matplotlib.pyplot as plt
from polynomialCodes import A1_HX, A1_HZ, A2_HX, A2_HZ, A3_HX, A3_HZ, A4_HX, A4_HZ, A5_HX, A5_HZ,  A6_HX, A6_HZ
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
    #print(cofactorMatrix)
    #for i in range(cofactorMatrix.shape[0]):
    #    for j in range(cofactorMatrix.shape[1]):
    #        print(isinstance(cofactorMatrix[i,j], Basic))
    cofactorsList = []
    for i in range(matrix.shape[0]):
        newRow = []
        for j in range(matrix.shape[1]):
            minor = matrix.copy() # Not very efficient, but good enough for one time use.
            minor.col_del(j)
            minor.row_del(i)
            newRow.append(symbolicDeterminant(minor))
            #print(f"For i=={i} , j=={j}, isinstance: {isinstance(detOfMinor, Basic)}")
            ##print(cofactorMatrix[i,j])
            #cofactorMatrix.subs(cofactorMatrix[i,j], detOfMinor)# (symbolicDeterminant(minor), cofactorMatrix[i,j])
        cofactorsList.append(newRow)
    return cofactorsList


def binaryGaussianEliminationOnRows(matrix):
    matrixInverse = np.eye(matrix.shape[0], dtype = np.bool)
    rank = 0
    for k in range(matrix.shape[1]):
        # Find the first row index i, such that row k + i has non zero element at column k
        for i in range(0, matrix.shape[0] - k, 1):
            if matrix[k + i, k] != 0:
                #print(i)
                break
        # Treat the case where no such row was found
        
        # If there is a row i with matrix[i,k] !=0, then subtract it from every row i+1+j below it where matrik[i+1+j,k] !=0
        for j in range(1, matrix.shape[0] - i - k, 1):
            if matrix[k + i + j, k] != 0:
                #print(f"Canceling row k + i + j =={k+i+j}")
                matrix[k + i + j, :] ^= matrix[k + i, :]
                matrixInverse[k + i + j, :] ^= matrix[k + i, :]
        # Now pivot:
        temp = copy.copy(matrix[k, :])
        matrix[k, :] = matrix[k + i, :]
        matrix[k + i, :] = copy.copy(temp)
        temp = copy.copy(matrixInverse[k,:])
        matrixInverse[k,:] = matrixInverse[k + i,:]
        matrixInverse[k + i, :] = copy.copy(temp)

    return matrix, matrixInverse, rank

    