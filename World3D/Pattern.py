import numpy as np
from uuid import uuid4
from World3D import Primitive, mat_mul

# Define Patterns
class Pattern(Primitive):
    """ Patterns take callable, single-value functions (such as Blueprint and Pattern),
        and, initialized with certain matricies, return new regions. """

    def __init__(self, P, W, Wr, I):
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
        
        # Get sizes from inputs
        self.OUT_SIZE = m = P.SIZE+7                                            # An important size for the matricies in this class
        self.SIZE = n = W[1].shape[1]                                           # Size of input is determined by x dimension of
        
        # Check matrix sizes
        assert all([w.shape == (n,m) for w in W[1:]]), "all(W[1:]) must be shape (n,m)."
        assert all([w.shape == (n,n) for w in Wr[1:]]), "all(Wr[1:]) must be shape (n,n)."
        assert all([i.shape == (1,n) for i in I[1:]]), "all(I[1:]) must be shape (1,n)." 
        assert W[0].shape == (m,),  "W[0] must be shape (m,)."
        assert Wr[0].shape == (n,),  "Wr[0] must be shape (n,)."
        assert I[0].shape == (1,), "I[0] must be int or size 1 array." 

        # Set Parameters
        self.P = P                                                              # Set elements
        self.W = W                                                              # Set weights
        self.Wr = Wr                                                            # Set recursive weights
        self.I                                                                  # Set iteration number elements
        
    def __call__(self, x):
        x = np.array(x)
        out = self._call_once(x)                                                # Call the first time
        z = mat_mul(x, self.I)[0]                                               # Get the number of times to loop
        for i in np.arange(z):                                                  # Loop z number either
            x = mat_mul(x, self.Wr)                                             # Modify x for next iteration
            out += self._call_once(x)                                           # Run next iteration and add to out
        return out
        
    def _call_once(self, x):
        """ Given some array x, shape (n,), returns a Blueprint by summing P1 and P2
            recursively over some number of iterations. """
        v = mat_mul(x, self.W)                                                  # Get the array in a shape P1 and P2 can accept
        x0 = v[:self.OUT_SIZE-7]
        y0 = v[self.OUT_SIZE-7:]
        return self.P(x0)(y0)                                                   # Get the region each pattern returns

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
        