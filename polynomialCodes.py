"""
This file implements codes from the paper "Degenerate Quantum LDPC Codes With Good Finite Length
Performance" by Pavel Panteleev and Gleb Kalachev

For polynomial arithmetic, we use the reedSolomon project: https://github.com/Omer-Sella/reedSolomon
or:
git@github.com:Omer-Sella/reedSolomon.git

Then set an environment variable REEDSOLOMON to the root directory of the project.
"""
import os, sys
reedSolomonProjectDir = os.environ.get('REEDSOLOMON')
if reedSolomonProjectDir == None: 
     raise("Please set the REEDSOLOMON environment variable to the root directory of the reedSolomon project")
sys.path.insert(0, reedSolomonProjectDir)
from arithmetic import polynomial
from scipy.linalg import circulant
import numpy as np

"""
Generalized bicycle (GB) codes. The matrices
A and B have form A = (a(x)), B = (b(x)), so
here we specify the polynomials a(x), b(x), and the
circulant size l.
"""
#A1 [[254, 28, d]] code (l = 127), 14 <=d<= 20.
#a(x) = 1 + x15 + x20 + x28 + x66,
#b(x) = 1 + x58 + x59 + x100 + x121.
a_254_28 = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
b_254_28 = [1] + [0]*57 + [1,1] + [0]*40 + [1] + [0]*20 + [1]
a_254_28_padded = a_254_28 + [0]*(127 - len(a_254_28))
A1_A = circulant(a_254_28_padded).transpose()
b_254_28_padded = b_254_28 + [0]*(127 - len(b_254_28))
A1_B = circulant(b_254_28_padded).transpose()
A1_HX = np.hstack((A1_A, A1_B))
A1_HZ = np.hstack((A1_B.transpose(), A1_A.transpose()))
#a_254_28.reverse()
#a = polynomial(a_254_28)
#a.printValues()
#b_254_28.reverse()
#b = polynomial(b_254_28)
#b.printValues()

#A2 [[126, 28, 8]] code (l = 63).
#a(x) = 1 + x + x14 + x16 + x22,
a_126_28_8 = [1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1]
a_126_28_8_padded = a_126_28_8 + [0]*(63 - len(a_126_28_8))
A2_A = circulant(a_126_28_8_padded).transpose()
#b(x) = 1 + x3 + x13 + x20 + x42.
b_126_28_8 = [1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
b_126_28_8_padded = b_126_28_8 + [0]*(63 - len(b_126_28_8))
A2_B = circulant(b_126_28_8_padded).transpose()
#a_126_28_8.reverse()
#b_126_28_8.reverse()
#b = polynomial(b_126_28_8)
#b.printValues()
A2_HX = np.hstack((A2_A, A2_B))
A2_HZ = np.hstack((A2_B.transpose(), A2_A.transpose()))

# A3) [[48, 6, 8]] code (l = 24).
# a(x) = 1 + x2 + x8 + x15,
a_48_6_8 = [1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1]
a_48_6_8_padded = a_48_6_8 + [0]*(24 - len(a_48_6_8))
A3_A = circulant(a_48_6_8_padded).transpose()
# b(x) = 1 + x2 + x12 + x17.
b_48_6_8 = [1,0,1] + [0]*9 + [1] + [0]*4 + [1]
b_48_6_8_padded = b_48_6_8 + [0]*(24 - len(b_48_6_8))
A3_B = circulant(b_48_6_8_padded).transpose()

#a_48_6_8.reverse()
#a = polynomial(a_48_6_8)
#a.printValues()
#b_48_6_8.reverse()
#b = polynomial(b_48_6_8)
#b.printValues()
A3_HX = np.hstack((A3_A, A3_B))
A3_HZ = np.hstack((A3_B.transpose(), A3_A.transpose()))

# A4) [[46, 2, 9]] code (l = 23).
# a(x) = 1 + x5 + x8 + x12,
a_46_2_9 = [1,0,0,0,0,1,0,0,1,0,0,0,1]
a_46_2_9_padded = a_46_2_9 + [0]*(23 - len(a_46_2_9))
A4_A = circulant(a_46_2_9_padded).transpose()
# b(x) = 1 + x + x5 + x7.
b_46_2_9 = [1,1,0,0,0,1,0,1]
b_46_2_9_padded = b_46_2_9 + [0]*(23 - len(b_46_2_9))
A4_B = circulant(b_46_2_9_padded).transpose()

A4_HX = np.hstack((A4_A, A4_B))
A4_HZ = np.hstack((A4_B.transpose(), A4_A.transpose()))
#a_46_2_9.reverse()
#a = polynomial(a_46_2_9)
#a.printValues()

#b_46_2_9.reverse()  
#b = polynomial(b_46_2_9)
#b.printValues()

# A5) [[180, 10, d]] code (l = 90), 15 <= d <= 18.
# a(x) = 1 + x28 + x80 + x89,
a_180_10 = [1] + [0]*27 + [1] + [0]*51 + [1] + [0]*8 + [1]
a_180_10_padded = a_180_10 + [0]*(90 - len(a_180_10))
A5_A = circulant(a_180_10_padded).transpose()
# b(x) = 1 + x2 + x21 + x25.
b_180_10 = [1,0,1] + [0]*18 + [1] + [0]*3 + [1]
b_180_10_padded = b_180_10 + [0]*(90 - len(b_180_10))
A5_B = circulant(b_180_10_padded).transpose()

A5_HX = np.hstack((A5_A, A5_B))
A5_HZ = np.hstack((A5_B.transpose(), A5_A.transpose()))
#a_180_10.reverse()
#a = polynomial(a_180_10)
#a.printValues()

#b_180_10.reverse()
#b = polynomial(b_180_10)     
#b.printValues()

#A6 [[900, 50, 15]] code (l = 450).
# a(x) = 1 + x97 + x372 + x425,
a_900_50_15 = [1] + [0]*96 + [1] + [0]*274 + [1] + [0]*52 + [1]
a_900_50_15_padded = a_900_50_15 + [0]*(450 - len(a_900_50_15))
A6_A = circulant(a_900_50_15_padded).transpose()
# b(x) = 1 + x50 + x265 + x390.
b_900_50_15 = [1] + [0]*49 + [1] + [0]*214 + [1] + [0]*124 + [1]
b_900_50_15_padded = b_900_50_15 + [0]*(450 - len(b_900_50_15))  
A6_B = circulant(b_900_50_15_padded).transpose()

A6_HX = np.hstack((A6_A, A6_B))
A6_HZ = np.hstack((A6_B.transpose(), A6_A.transpose()))

#a_900_50_15.reverse()    
#a = polynomial(a_900_50_15)
#a.printValues()

#b_900_50_15.reverse()
#b = polynomial(b_900_50_15)
#b.printValues()

"""
B. Generalized hypergraph product (GHP)
codes. Here the matrix B is diagonal; B = b(x)I_n,
where I_n is the n×n identity matrix over the ring
F^{<l>}_2.
"""
#B1 [[882, 24, d]] code, 18 <= d <= 24. The matrices
#HX and HZ are (3,6)-regular (l = 63).
#A =
#[
#x27 0 0 0 0 1 x54
#x54 x27 0 0 0 0 1
#1 x54 x27 0 0 0 0
#0 1 x54 x27 0 0 0
#0 0 1 x54 x27 0 0
#0 0 0 1 x54 x27 0
#0 0 0 0 1 x54 x27
#]
#B = (1 + x + x6)I7

"""
Hypergraph product (HP) codes. Each hy-
pergraph product code in our simulations is con-
structed from a single cyclic code dened by its
parity polynomial h(x) and the length l.
"""
# C1) [[7938, 578, 16]] code. The matrices HX and
# HZ are (5,10)-regular, and we have:
# ` = 63, h(x) = 1 + x3 + x34 + x41 + x57.
# C2) [[1922, 50, 16]] code. The matrices HX and
# HZ are (3,6)-regular, and we have:
# ` = 31, h(x) = 1 + x2 + x5.