from qecc.funWithMatrices import binaryGaussianEliminationOnRows
import numpy as np



def sameCode(codeA, codeB):
    """Two CSS codes are equal iff their X-checks span the same row space and likewise Z.

    rank([H1; H2]) == rank(H1) == rank(H2) for both the X and Z parity matrices (GF(2)).
    """
    Hx1, Hz1 = codeA
    Hx2, Hz2 = codeB
    _, _, rankHx1 = binaryGaussianEliminationOnRows(np.array(Hx1, copy=True))
    _, _, rankHx2 = binaryGaussianEliminationOnRows(np.array(Hx2, copy=True))
    _, _, rankHz1 = binaryGaussianEliminationOnRows(np.array(Hz1, copy=True))
    _, _, rankHz2 = binaryGaussianEliminationOnRows(np.array(Hz2, copy=True))
    _,_ , rankHxStacked = binaryGaussianEliminationOnRows(np.array(np.vstack([Hx1,Hx2]), copy = True))
    _,_ , rankHzStacked = binaryGaussianEliminationOnRows(np.array(np.vstack([Hz1,Hz2]), copy = True))
    
    if not (rankHxStacked == rankHx1 == rankHx2):
            return False
    if not (rankHzStacked == rankHz1 == rankHz2):
            return False
    return True

def codeWeights(parityMatrix):
       return np.nonzero(parityMatrix)


