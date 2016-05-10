from uuid import uuid4
import numpy as np

KEY = {'AIR': 0, 'STONE': 1, 'STAIR': 2}
LOOKUP = dict((v, k) for k, v in KEY.items())

def mat_mul(x, W):
    out = W[0]
    for i, w in enumerate(W[1:]):
        out += np.dot(w, x**i)
    return out

class Primitive(object):
    SIZE = None
    def __init__(self):
        self.ID = uuid4()
        
    def __call__(self, x):
        raise NotImplementedError()

    def __len__(self):
        return self.SIZE
    
    # Control Required Attributes
    def __getattr__(self, name):
        """ SIZE & ID must be instantiated. """
        if name in ('ID','SIZE'):
            if getattr(None, self, name) is None:
                raise NotImplementedError()
        return getattr(None, self, name)
        
    # Immutability
    def __hash__(self):
        
        return hash(self.ID)
        
    def __eq__(self, other):
        """ Checks all values in each item to see if they are the same. """
        out = True
        for name in self.__dict__:
            out = out and np.all(getattr(None,self,name) == getattr(None,other,name))
        return out
        
    def __setattr__(self, name, val):
        """ Controlls immutability, locks values after they have been set. """
        if name not in self.__dict__ or self.__dict__[name] is None:
            self.__dict__[name] = val
        else:
            raise IndexError("{0} is immutable. {1} has already been set.".format(repr(self), name))

class Cube(Primitive):
    SIZE = 7
    def __init__(self, block_id=0):
        self.ID = uuid4()
        self.BLOCK_ID = block_id
        
    def __call__(self, x):
        """ Params: x[0:3]: Size vector; The size of the cube in 3D
                    x[3:6]: Rotation vector.
                    x[6]:   scalar; The layer the cube exists in (used in merging).Rotation
                    block_id: The block ID of the cube. """
        dx, dy, dz, r, l = x[0:3], x[3:6], x[6]                                 # Breakdown the input
        R = np.ones((dx, dy, dz, 3)) * np.array(r)                              # Set the rotation matrix using r vector
        L = np.full((dx, dy, dz), fill_value = float(l))                        # Initialize layer with layer value
        V = np.full((dx, dy, dz), fill_value = float(self.BLOCK_ID))            # Initialize the value with block_id
        return Region(V = V, R = R, L = L, x0 = (x0, y0, z0))                   # Create the region

if __name__ == "__main__":
    # Create all known patterns
    PATTERNS = {}
    
    # Get a list of all possible cubes
    PATTERNS += dict((key+"_Cube", Cube(block_id = val)) for key, val in KEY.items())
    
    # Create a stair piece, input_array = [x0, y0, z0, height, width]
    P1 = PATTERNS['STONE_Cube']
    P2 = PATTERNS['STAIR_Cube']
    
    # index:       dx    dy    dz    r0    r1    r2    l
    Wx = np.array([[0,1],[0,0],[1,0],[0,0],[0,0],[0,0],[0,0],
                   [0,1],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]])
    Wy = np.array([[0,0],[0,0],[0,0],
                   [0,0],[0,0],[0,0]])
    
    # index:      dx,dy,dz,r0,r1,r2,l              
    Bx = np.array([0, 0, -1, 0, 0, 0, 0,
                   0, 0, 1,  0, 0, 0, 0])
    By = np.array([0, 0, 0, 1, 0, 0])
    
    PATTERNS['Stairs_Piece'] = Pattern(P1,P2,W,B)

    # Create Stairs, input_array = [height, width]
    P1 = PATTERNS['Stairs_Piece']
    P2 = PATTERNS['Stairs_Piece']
    # Generate Matricies
    Wx = np.array([[1,0],[0,1],
                   [1,0],[0,1]])          # 
    Wy = np.array([[1,0],[0,0],[0,0],
                   [1,0],[0,0],[0,0]])    # The x axis shift will be perportional to the height
    Bx = np.array([0,0,-1,0])             # We will move P2's height down 1
    By = np.array([0,0,0,-1,0,0])         # We will move P2's location left 1
    Wr = np.array([[1,0],[0,1]])          # No perportional change in parameters recursively
    Br = np.array([-1,0])                 # Height is reduced by one over recursion
    I  = np.array([[1,0]])                # Iterations perportional to height
    
    PATTERNS['Stairs'] = Pattern(W = Wx, B = Bx, Wy = Wy, By = By, Wr = Wr, Br = Br, I = I, i = 0)
