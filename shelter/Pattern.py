import numpy as np
from multiprocessing import pool
from uuid import uuid4
from World3D import *

# Define Elements
class Element(object):
    """ Elements are wrappers for regions that can be instantiated by a single len(n) array. """
    SIZE = None
    def __call__(self, x):
        raise NotImplemented
        
# Define Primitive Elements
class CubeElement(Element):
    def __init__(self, block_id):
        self.R = Cube
        self.block_id = block_id
        self.SIZE = 10
    def __call__(self, x):
        x, sz = np.array(x), self.R.SIZE
        assert(x.shape == (self.SIZE,), "X must be shape (self.SIZE,)")
        return self.R(x = x[0:sz], block_id = self.block_id, x0 = x[sz,sz+3], r = x[sz+3:sz+3*2], l = x[sz+3*2])

class BlockElement(Element):
    def __init__(self, block_id):
        self.R = Block
        self.block_id = block_id
        self.SIZE = 7
    def __call__(self, x):
        x, sz = np.array(x), self.R.SIZE
        assert(x.shape == (self.SIZE,), "X must be shape (self.SIZE,)")
        return self.R(x = x[0:0], block_id = self.block_id, x0 = x[0:3], r = x[3:6], l = x[6])
        
class Mold(Element):
    def __init__(self, pattern):
        self.pattern = pattern
    def __call__(self, x):
        a = pattern.SIZE
        out = self.pattern(x[0:a])
        out = out.move(x[a:a+3])
        out = out.rotate(x[a+3:a+6])
        out = out.flatten(x[a+6])
        return out
        
# Define Patterns
class Pattern():
    """ Patterns take elements and matricies and return new elements. """
    MAX_ITER = 100
    def __init__(self, E1, E2, W, B, N, n, Wr = None, Br = None):
        self.OUT_SIZE = E1.SIZE + E2.SIZE                                       # An important size for the matricies in this class
        self.SIZE = W.shape[1]                                                  # Size of input is determined by x dimension of W
        
        # Check all matrix sizes
        W, B, N = np.array(W), np.array([B]).T, np.array([N])
        assert(W.shape[0] == self.OUT_SIZE), "W must have a y dimension the same size as the length of the expected input vector for __call__.")
        assert(B.shape == (self.OUT_SIZE,)), "Biases must be vectors of the same size as the length of the expected input vector for __call__.")
        assert(N.shape == (self.OUT_SIZE,)), "Number matrix must be the same size as the length of the expected input vector"
        if Wr is not None and Br is not None:
            self.recursive = True                                               # Recursive elements require Wr and Br
            Wr, Br = np.array(Wr), np.array([Br]).T
            assert(Br.shape == B.shape, "Biases must be vector of the same size as the length of the expected input vector for __call__.") 
            assert(Wr.shape == (self.OUT_SIZE,self.OUT_SIZE), "Wr must either be a matrix of the same size as the length of the expected input vector for __call__, squared.")                       # Handle Wr as a matrix
        else:
            self.recursive = False                                              # Recursive elements require Wr and Br
        
        # Set Matricies and Elements    
        self.E1, self.E2 = E1, E2                                               # Set elements
        self.W, self.B = W, B                                                   # Set weights
        self.Wr, self.Br = Wr, Br                                               # Set recursive weights
        self.N, self.n = N, n
        
    def __call__(self, x):
        a, b = self.E1.SIZE, self.E2.SIZE
        x = np.dot(W,np.array([x]).T)+B
        A = self.E1(x[:,0:a])
        B = self.E2(x[:,a:b+a])
        out = A + B
        
        # Set up recursive loop
        n = np.dot(self.N, x) + self.n
        while self.recursive and n >= 0:
            x = np.dot(Wr,x)+Br
            out += self.E1(x[:,0:a])
            out += self.E2(x[:,a:b+a])
            n -= 1
        
        return Mold(out)
    
# Create some examples
def hollow(d3, x0 = (0,0,0), r = (0,0,0), l = 1.):
    """ d3 is your size vector, shape (3,) """
    assert(np.all(d3 > 2), 'A Hollow Cube must be at least size 2 in all dimensions')
    rock = cube(block_id = KEY['stone'], d3 = d3, x0 = x0, r = r, l = 1+l)
    air  = cube(block_id = KEY['air'], d3 = d3-2, x0 = x0+1, r = r, l = 0+l)
    out = air + rock
    out.SIZE = 3
    
def stairs(d2, x0 = (0,0,0), r = (0,0,0), l = 1.):
    """ d2 is the height [0] and width [1] of the stairs """
    assert(np.all(d2 > 0), 'Stairs must at least be height and width 1.')
    stair = cube(block_id = KEY['stair'], d3 = (1,d2[1],1), x0 = (x0[0]+d2[0]-1, x0[1], x0[2]), r = r, l = l)
    foundation = cube(block_id = KEY['wall'], d3 = (1,d2[1],d2[0]-1)), x0 = (x0[0], x0[1], x0[2]+1)
    out = stair + foundation
    try:
        out += stairs(d2 = (d2[0]-1, d2[1]), x0 = (x0[0]+1, x0[1]+1, x0[2]), r = r, l = l)
    except AssertionError:
        out.SIZE = 2
        return out
 
# Modify them to use Pattern class
def stairs2DElement():
    stairElement = BlockElement(block_id = KEY['STAIR'])
    stoneElement = CubeElement(block_id = KEY['STONE'])

    # Transform [1] into [0,0,0,0,0,0,0]+[0,1,0,0,0,0,0,0,0,0] with W1
    # Transform [2] into [0,0,0,0,0,0,0]+[0,2,0,0,0,0,0,0,0,0] with W1
    W = np.array([[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]]).T
    
    # Transform [0,0,0,0,0,0,0]+[0,1,0,0,0,0,0,0,0,0] into [0,0,0,0,0,0,1]+[1,1,1,0,1,0,0,0,0,1] with B1
    # Transform [0,0,0,0,0,0,0]+[0,2,0,0,0,0,0,0,0,0] into [0,0,0,0,0,0,1]+[1,2,1,0,1,0,0,0,0,1] with B1
    B = np.array([0,0,0,0,0,0,1,1,0,1,0,1,0,0,0,0,1])
    N, n = np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]), 0
    
    # Transform [0,0,0,0,0,0,1]+[1,1,1,0,1,0,0,0,0,1] into [0,1,1,0,0,0,1]+[1,0,1,0,2,1,0,0,0,1] with W2 + B2
    # Transform [0,0,0,0,0,0,1]+[1,2,1,0,1,0,0,0,0,1] into [0,1,1,0,0,0,1]+[1,1,1,0,2,1,0,0,0,1] with W2 + B2
    # Transform [0,1,1,0,0,0,1]+[1,1,1,0,2,1,0,0,0,1] into [0,2,2,0,0,0,1]+[1,0,1,0,3,2,0,0,0,1] with W2 + B2
    Wr = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])*np.eye(17)
    Br = np.array([0,1,1,0,0,0,0]+[0,-1,0,1,1,0,0,0,0,0])
    
    out = Pattern(stairElement, stoneElement, W, B, N, n, Wr, Br)
    return out
    
def stairs3DElement():
    stairs2DE = stairs2DElement(x0 = x0, r = r, l = l)
    
    W = np.array([[0,0,0,0,0,0,0]+[0,0,0,0,0,0,0],[1,1,1,1,1,1,1]+[1,1,1,1,1,1,1]]).T
    B = np.array([0,0,0,0,0,0,0]+[1,0,0,0,0,0,0]+[0,0,0,0,0,0,0]+[0,0,0,0,0,0,0])
    Wr = np.array([1,1,1,1,1,1,1]+[1,1,1,1,1,1,1]+[1,1,1,1,1,1,1]+[1,1,1,1,1,1,1])*np.eye(14*2)
    Br = np.array([1,0,0,0,0,0,0]+[1,0,0,0,0,0,0]+[0,0,0,0,0,0,0]+[0,0,0,0,0,0,0])
    N, n = np.array([])

    out = Pattern(stairs2DE, stairs2DE, W, B, N, n, Wr, Br)
    return out
    
if __name__ == "__main__":
    a = cube(1, dx = [2,2,2])
    b = cube(2, dx = [2,2,2], x0 = [2,1,1])
    c = block(3, x0 = [0,3,0])
    print(a)
    print(b)
    print(c)
    print((a+b)+c)
    
# class Pattern(object):
#     # Define Globals
#     GREATER_INDEX, LESSER_INDEX, EQUALS_INDEX = 0, 1, 2
    
#     def __init__(self, block_options):
#         """ Block options is a 2D array that lists blocks by their number in axis 0, and lists options for that block in axis 1 """
#         # Create blocks list
#         self.blocks, self.Nobjs = np.array(block_options), len(block_options)
        
#         # The location poly in 4D space. shape = (#obj, [x,y,z,l], [x,*poly])
#         # The size poly in 3D space. shape = (#obj, [dx,dy,dz], [x,*poly])
#         # The rotation poly in 3D space. shape = (#obj, [rx,ry,rz], [x,*poly])
#         # Combine into self._X
#         self._X  = np.zeros([self.Nobjs, 10, 3])
        
#         # Similarly, there are constraints
#         self.GREATER = -np.ones([self.Nobjs, 10, 3])
#         self.LESSER  =  np.ones([self.Nobjs, 10, 3])
#         self.EQUALS  =  np.ones([self.Nobjs, 10, 3])
        
#         # And we may select any of them we wish by signaling True in axis 3, following:
#         # shape = (#obj, dim, poly, [GREATER, LESSER, EQUALS])
#         self.SEL     =  np.full([self.Nobjs, 10, 3, 3], fill_value=False)
        
#     def matrix_constraint(value, obj_n, dim_n, poly_n, sel_index, activate = True):
#         """ Can be used to set many constraints at once, given a array value. """
#         self.GREATER[obj_n, dim_n, poly_n] = value[:,:,:,self.GREATER_INDEX]
#         self.LESSER[obj_n, dim_n, poly_n]  = value[:,:,:,self.LESSER_INDEX]
#         self.EQUALS[obj_n, dim_n, poly_n]  = value[:,:,:,self.EQUALS_INDEX]
#         self.SEL[obj_n, dim_n, poly_n, sel_index]  = activate
    
#     def constraint(value, obj_n, dim_n, poly_n, sel_index, activate = True):
#         """ Used to set just one type of constraint at a time. """
#         if sel_index == self.GREATER_INDEX:
#             self.GREATER[obj_n, dim_n, poly_n] = value
#         elif sel_index == self.LESSER_INDEX:
#             self.LESSER[obj_n, dim_n, poly_n]  = value
#         elif sel_index == self.EQUALS_INDEX:
#             self.EQUALS[obj_n, dim_n, poly_n]  = value
#         else:
#             raise KeyError("Selection Index {0} not an option.".format(sel_index))
            
#         self.SEL[obj_n, dim_n, poly_n, kind_index]  = activate
            
#     def __copy__(self):
#         out = Pattern(deepcopy(self.blocks))
#         out._X = deepcopy(self._X)
#         out.GREATER = self.GREATER
#         out.LESSER  = self.LESSER
#         out.EQUALS  = self.EQUALS
#         out.SEL     = self.SEL
#         out.WORLD   = self.WORLD
#         return out