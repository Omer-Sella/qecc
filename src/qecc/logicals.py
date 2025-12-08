"""
Compute the logical operators of a CSS code given its stabilizer generators.


"""

from qecc import funWithMatrices
import copy
import numpy as np

def computeLogicals(stabilizerGeneratorsX, stabilizerGeneratorsZ):
    """
    Given the stabilizer generators of a CSS code, compute its logical operators.

    Input:
        stabilizerGeneratorsX (np.ndarray): The X stabilizer generators - not assumed to be in reduced form.
        stabilizerGeneratorsZ (np.ndarray): The Z stabilizer generators - not assumed to be in reduced form.
    Output:
        logicalOperatorsX (np.ndarray): The X logical operators - in reduced form, but could contain a component from stabilizerGeneratorsX.
        logicalOperatorsZ (np.ndarray): The Z logical operators - in reduced form, but could contain a component from stabilizerGeneratorsZ.
    
    We first reduce all input matrices to their row echelon form.

    For a CSS code with parity-check matrices $H_X$ and $H_Z$ (over $F_2$), corresponding to stabilizerGeneratorsX and stabilizerGeneratorsZ we first find logical $Z$ operators. 
    The $Z$ logical operators are those $Z$ operators that commute with all $X$ stabilizers (i.e., they are in the normalizer of S), but are not in the span of the $Z$ stabilizers, i.e. in S.
    We first look for all $Z$ operators that commute with all $X$ stabilizers by solving the equation $H_X z = 0$ over $F_2$ and obtaining a set of basis vectors for this space.
    Some of them are $X$-stabilizers, some are logical $X$ operators. (Z stabilizers, Z operators)
    To check if a logical $Z$ operator, v, is in the span of the matreix $H_X$ == stabilizerGeneratorsX, we test whether the rank of stabilizerGeneratorsX is strictly smaller than that of np.vstack((stabilizerGeneratorsX, v)). If yes, it means that v is in the row space of stabilizerGeneratorsX.
    Any v that survived this test is a logical $Z$ operator. 
    """
    
    stabilizerGeneratorsX, stabilizerGeneratorsXInverse, stabilizerGeneratorsXrank  = funWithMatrices.binaryGaussianEliminationOnRows(copy.copy(stabilizerGeneratorsX))
    stabilizerGeneratorsZ, stabilizerGeneratorsZInverse, stabilizerGeneratorsZrank  = funWithMatrices.binaryGaussianEliminationOnRows(copy.copy(stabilizerGeneratorsZ))

    logicalOperatorsX = funWithMatrices.solveHomogenicBinaryLinearSystem(stabilizerGeneratorsZ)
        
    newLogicalOperatorsX = []
    for i in range(logicalOperatorsX.shape[0]):
        testMatrix = np.vstack( (stabilizerGeneratorsX, logicalOperatorsX[i,:]) )
        _, _, testRank = funWithMatrices.binaryGaussianEliminationOnRows(copy.copy(testMatrix))
        if testRank > stabilizerGeneratorsXrank:
            # This row is in the span of the X stabilizers, remove it
            newLogicalOperatorsX.append(logicalOperatorsX[i,:])
    newLogicalOperatorsZ = []
    
    logicalOperatorsZ = funWithMatrices.solveHomogenicBinaryLinearSystem(stabilizerGeneratorsX)
    for i in range(logicalOperatorsZ.shape[0]):
        testMatrix = np.vstack( (stabilizerGeneratorsZ, logicalOperatorsZ[i,:]) )
        _, _, testRank = funWithMatrices.binaryGaussianEliminationOnRows(copy.copy(testMatrix))
        if testRank > stabilizerGeneratorsZrank:
            # This row is in the span of the stabilizers, remove it
            newLogicalOperatorsZ.append(logicalOperatorsZ[i,:])

    return np.array(newLogicalOperatorsX), np.array(newLogicalOperatorsZ)

