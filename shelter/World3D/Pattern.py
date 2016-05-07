import numpy as np
from uuid import uuid4
from World3D import Primitive

# Define Patterns
class Pattern(Primitive):
    """ Patterns take callable, single-value functions (such as Blueprint and Pattern),
        and, initialized with certain matricies, return new regions. """

    def __init__(self, P1, P2, W, B, Wr = None, Br = None, I = None, i = 0, Wy = None, By = None):
        """ m = P1.SIZE[0] + P2.SIZE[0]
            n = self.SIZE[0]
            W:  shape (n,m)
            B:  shape (m,1)
            Wr: shape (n,n)
            Br: shape (n,1)
            I:  shape (1,n)
            i:  int """
        
        # Set ID
        self.ID = uuid4()
        
        # Check types
        W, B = np.array(W), np.array(B)
        assert(isinstance(n, int), "i must be an integer.")
        
        # Get sizes from inputs
        m = P1.SIZE[0] + P2.SIZE[0]                                             # An important size for the matricies in this class
        n = W.shape[1]                                                          # Size of input is determined by x dimension of W
        
        # Check matrix sizes
        assert(W.shape  = (n,m), "W must be shape (n,m).")
        assert(B.shape  = (m,),  "B must be shape (m,).")
        
        # Handle recursive elements
        if Wr is not None and Br is not None and I is not None:
            I = np.array(I)
            assert(Wr.shape = (n,n), "Wr must be shape (n,n).")
            assert(Br.shape = (n,),  "Br must be shape (n,).")
            assert(N.shape  = (1,n), "N must be shape (1,n).")
            self.RECURSIVE = True
        else:
            self.RECURSIVE = False
        
        # Set Parameters
        self.P1, self.P2 = P1, P2                                               # Set elements
        self.W, self.B = W, B                                                   # Set weights
        self.Wr, self.Br = Wr, Br                                               # Set recursive weights
        self.I, self.i = I, i                                                   # Set iteration number elements
        
        if Wy is not None and By is not None:
            Wy, By = np.array(Wy), np.array(By)
            my, ny = P1.SIZE[1] + P2.SIZE[1], Wy.shape[1]
            assert(Wy.shape = (ny, my))
            assert(By.shape = (my, ))
        else:
            my, ny = 0, 0
        self.OUT_SIZE = (m, my)
        self.SIZE = (n,ny)
        
        self.Wy, self.By = Wy, By                 # Set the y matricies
        
    def __call__(self, x, y = None):
        x = np.array(x)
        out = self._call_once(x,y)
        if self.RECURSIVE:                                                      # If this pattern is recursive
            z = (np.dot(self.I,input_array)+self.i)[0]                          # Get the number of times to loop
            for i in np.arange(z):                                              # Loop z number either
                out += self._call_once(x,y)
        return out
        
    def _call_once(self, x, y = None):
        """ Given some array x, shape (n,), returns a Blueprint by summing P1 and P2
            recursively over some number of iterations. """
        
        # Handle y if not None
        if y is not None:
            y = np.array(y)
            in_y = np.dot(self.Wy, y) + self.By
            y1, y2 = in_y[:self.P1.SIZE[1]], in_y[self.P1.SIZE[1]:]
        else:
            y1, y2 = None
            
        # Handle x
        in_x = np.dot(self.W, x) + self.B                                       # Get the array in a shape P1 and P2 can accept
        x1, x2 = in_x[:self.P1.SIZE[0]], in_x[self.P1.SIZE[0]:]
        r1 = self.P1(x1, y1)                                                    # Get the region each pattern returns
        r2 = self.P2(x2, y2)                                                    # --
        return r1 + r2                                                          # Sum them together

    def __repr__(self):
        """ Prints this Pattern's ID """
        return "<Pattern ID: {0}>".format(self.ID)
        
    def __hash__(self):
        """ Hashes the ID of this object. """
        return hash(self.ID)
        
    def __eq__(self, other):
        """ Checks that all tensors are equal """
        return self.P1 == other.P1 and self.P2 == other.P2 and \
                np.all(self.W == other.W) and \
                np.all(self.B == other.B) and \
                np.all(self.Wr == other.Wr) and \
                np.all(self.Br == other.Br) and \
                np.all(self.I == other.I) and \
                np.all(self.i == other.i)
    
    # Pickling and saving
    # FIXME: Implement this
    def __getstate__(self):
        raise NotImplemented
        
    def __setstate__(self, state):
        raise NotImplemented