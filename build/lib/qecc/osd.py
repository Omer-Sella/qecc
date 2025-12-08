from src.qecc import funWithMatrices
import numpy as np
def osdDecoder(H, syndrome, coordinateReliabilities, upToRank = np.inf):
    """
    Recover error vector from syndrome using least reliable coordinates.
    
    Args:
        H: Parity check matrix
        syndrome: Calculated syndrome vector
        coordinareReliabilities: Vector of coordinate reliabilities (the lower the value, the less reliable the coordinate)
        
    Returns:
        Recovered error vector
    
    Note:
        No safety, meaning, the syndrome is assumed to be in the column span of H.
    """
    #
    # Step 1: get the permutation pi1 that, if applied to the indices of coordinateReliabilities, sorts coordiunateReliabilities in an ascending order
    pi1 = np.argsort(coordinateReliabilities)
    # Step 2: Reorder the columns of H according to pi1
    H_pi1 = H[:, pi1]
    # Step 3: Find the solution for the first r independent columns of H_pi1
    matrix, matrixInverse, rank = funWithMatrices.binaryGaussianEliminationOnRows(H_pi1)
    

    syndromeTransformed = matrixInverse[:,:len(syndrome)] @ syndrome % 2
    
    
    #if upToRank < rank:
    #    rank = upToRank
    _, linearlyIndependentIndices = np.unique(matrix[:,:rank], axis=1, return_index=True)
    linearlyIndependentIndices.sort()
    # The reliability score is the sum of the reliabilitires of the used coordinates
    reliabilityScore = np.sum(coordinateReliabilities[pi1[0:rank]])
    solution = np.zeros(H.shape[1], dtype=np.bool)
    solution[pi1[0:rank]] = syndromeTransformed[linearlyIndependentIndices]
    
    return solution, reliabilityScore