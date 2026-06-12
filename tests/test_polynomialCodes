def test_dimensionIsEquivalentToIBMPaper():
    """

    [[n,k,d]] NetEncoding   ℓ,m A B
                Rate r 
    [[72,12,6]] 1/12        6,6 x3+y+y2 y3+x+x2
    [[90,8,10]] 1/23        15,3 x9+y+y2 1+x2+x7
    [[108,8,10]] 1/27       9,6 x3+y+y2 y3+x+x2
    [[144,12,12]] 1/24      12,6 x3+y+y2 y3+x+x2
    [[288,12,18]] 1/48      12,12 x3+y2+y7 y3+x+x2
    [[360,12,≤24]] 1/60     30,6 x9+y+y2 y3+x25+x26
    [[756,16,≤34]] 1/95     21,18 x3+y10+y17 y5+x3+x19
    """

    import copy
    from qecc.polynomialCodes import codes
    from qecc.funWithMatrices import binaryGaussianEliminationOnRows

    

    def codeDimensionK(length,Hx,Hz):
        _,_, rankX = binaryGaussianEliminationOnRows(copy.copy(Hx))
        _,_, rankZ = binaryGaussianEliminationOnRows(copy.copy(Hz))    
        return length - rankZ - rankZ
    
    assert(codeDimensionK(codes["Hx_72_12_6"].shape[1], codes["Hx_72_12_6"], codes["Hz_72_12_6"]) == 12)
    assert(codeDimensionK(codes["Hx_90_8_10"].shape[1], codes["Hx_90_8_10"], codes["Hz_90_8_10"]) == 8)
    assert(codeDimensionK(codes["Hx_108_8_10"].shape[1], codes["Hx_108_8_10"], codes["Hz_108_8_10"]) == 8)
    assert(codeDimensionK(codes["Hx_144_12_12"].shape[1], codes["Hx_144_12_12"], codes["Hz_144_12_12"]) == 12)
    assert(codeDimensionK(codes["Hx_288_12_18"].shape[1], codes["Hx_288_12_18"], codes["Hz_288_12_18"]) == 12)
    assert(codeDimensionK(codes["Hx_360_12_24"].shape[1], codes["Hx_360_12_24"], codes["Hz_360_12_24"]) == 12)
    assert(codeDimensionK(codes["Hx_756_16_34"].shape[1], codes["Hx_756_16_34"], codes["Hz_756_16_34"]) == 16)

if __name__ == "__main__":
    test_dimensionIsEquivalentToIBMPaper()