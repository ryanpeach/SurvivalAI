import numpy as np
from uuid import uuid4
from World3D import *
from Pattern import mat_mul

class Blueprint(Primitive):
    """ A Blueprint adds and moves lists of callable objects which return regions. """
        
    def __init__(self, P, W):
        self.ID = uuid4()
        self.P = tuple(P)
        self.W = tuple(W)
        self.SIZE = W[0][1].shape[1]
        
    def __call__(self, x):
        x = np.array(x)
        i1, i2, i3 = 0, self.SIZE-7, self.SIZE
        v = mat_mul(x[i1:i3], self.W[0])
        out = self.P[0](v[i1:i2])(v[i2:i3])
        for p, w in zip(self.P[1:], self.W[1:]):
            i1, i2, i3 = i1 + self.SIZE, i2 + self.SIZE, i3 + self.SIZE
            v = mat_mul(x[i1:i3], w)
            out += o(x[i1:i2])(x[i2:i3])
        return out

    def __str__(self):
        """ Prints this Blueprint's self list. """
        return str(self.P)
        
    def __repr__(self):
        """ Prints this Blueprint's ID """
        return "<Blueprint ID: {0}>".format(self.ID)
        
    def __hash__(self):
        """ Hashes the ID of this object. """
        return hash(self.ID)
        
    def __eq__(self, other):
        """ Checks that all items in list are equal """
        return np.all([p0 == p1 for p0, p1 in zip(self.P, other.P)]) and \
                np.all([w0 == w1 for w0, w1 in zip(self.W, other.W)])
    
    # Pickling and saving
    # FIXME: Implement this
    def __getstate__(self):
        raise NotImplementedError()
        
    def __setstate__(self, state):
        raise NotImplementedError()