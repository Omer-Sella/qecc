import numpy as np

class gf4():
    def __init__(self, value):        
        if value < 0 or value > 3:
            raise ValueError("Value must be between 0 and 3 inclusive.")
        self.value = value
    
    def lsb(self):
        return self.value % 2
    def msb(self):
        return (self.value >> 1) & 1

    def __add__(self, other):
        if not isinstance(other, gf4):
            raise TypeError("Can only add two gf4 objects.")
        msb = (self.msb() + other.msb()) % 2
        lsb = (self.lsb() + other.lsb()) % 2
        return gf4(msb * 2 + lsb)
    
    def __mul__(self, other):
        if not isinstance(other, gf4):
            raise TypeError("Can only multiply two gf4 objects.")
        # \omega^2 = \omega + 1
        
        [msb, lsb] = [1,1] if self.msb() * other.msb() == 1 else [0,0]
        msb = (msb + self.lsb() * other.msb() + self.msb() * other.lsb() ) % 2
        lsb = (lsb + self.lsb() * other.lsb() )% 2
        return gf4(msb * 2 + lsb)
    
    def bar(self):
        return gf4(self.value) if self.msb() == 0 else gf4((self.msb() * 2 + (self.lsb() + 1) % 2))
        
    
    def tr(self, other = None):
        if other is not None:
            if not isinstance(other, gf4):
                raise TypeError("Trace can only be computed with another gf4 object.")
            return (self * other.bar()).tr()
        return (self + (self * self))

   
    def __repr__(self):
        if self.value == 0:
            return "0"
        elif self.value == 1:
            return "1"
        elif self.value == 2:
            return "ω"
        elif self.value == 3:
            return "(ω + 1)"
        else:
            return f"Invalid gf4 element with value {self.value}"
        
    def __eq__(self, other):
        if not isinstance(other, gf4):
            return False
        return self.value == other.value
    
    

def mul(a,b):  
    if not all(isinstance(a[i], gf4) for i in range(len(a))) or not all(isinstance(b[i], gf4) for i in range(len(b))):
        raise TypeError("Can only multiply two arrays of gf4 objects.")
    if len(a) != len(b):
        raise ValueError("Arrays must be of the same length.")
    result = gf4(0)
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

def add(a,b):
    if not all(isinstance(a[i], gf4) for i in range(len(a))) or not all(isinstance(b[i], gf4) for i in range(len(b))):
        raise TypeError("Can only add two arrays of gf4 objects.")
    if len(a) != len(b):
        raise ValueError("Arrays must be of the same length.")
    result = []
    for i in range(len(a)):
        result.append(a[i] + b[i])
    return result

def traceInner(a,b):
    if not all(isinstance(a[i], gf4) for i in range(len(a))) or not all(isinstance(b[i], gf4) for i in range(len(b))):
        raise TypeError("Can only compute inner product of two arrays of gf4 objects.")
    if len(a) != len(b):
        raise ValueError("Arrays must be of the same length.")
    result = gf4(0)
    for i in range(len(a)):
        result += a[i] * (b[i].bar())
    return result.tr()

def toarray(a):
    if not all(isinstance(a[i], gf4) for i in range(len(a))):
        raise TypeError("Can only convert an array of gf4 objects to a numpy array.")
    result = np.zeros(len(a), dtype=object)
    for i in range(len(a)):
        result[i] = a[i].value
    return result

    
    