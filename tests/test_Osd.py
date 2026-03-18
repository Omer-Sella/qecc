def test_osd1():
    import numpy as np
    from qecc.osd import osdDecoder 
    from qecc.polynomialCodes import A1_HX
    code = A1_HX.astype(np.int32)
    error = np.zeros(code.shape[1], dtype=np.int32)
    error[0]= True
    syndrome = code @ error %2
    coordinateReliabilities = np.ones(code.shape[1])
    coordinateReliabilities[0] = 0.1  # make coordinate 0 the least reliable
    solution, reliability = osdDecoder(code, syndrome, coordinateReliabilities)
    
    assert(np.all(solution  == error))

if __name__ == "__main__":
    test_osd1()