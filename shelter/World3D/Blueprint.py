import numpy as np
from uuid import uuid4
from World3D import Primitive

class Blueprint(object):
    """ A diagram accepts Regions as inputs, and stores them independent of location, rotation, and layer.
        They are immutable and callable to initialize Regions. """
        
    SIZE = 7    # They have a size of 7: 3 location dims, 3 rotation dims, 1 layer dim
    
    def __init__(self, V, R = None, L = None):
        """ Takes a value vector (V), and optional rotation (R) and layer (L) tensors, which will otherwise be set by default as zeros/ones.
            V: Value tensor;    shape: (n, m);    accepts: array
            R: Rotation tensor; shape: (n, m, 3); accepts: None, array.
            L: Layer tensor;    shape: (n, m);    accepts: None, int, array. """
            
        # Set properties
        self.V = np.array(V)
        self.shape = np.array(V.shape)
        self.ID = uuid4()
        
        # Default R is same shape as V with an extra dimension size 3
        if R is None:
            sx, sy, sz = V.shape
            self.R = np.zeros((sx,sy,sz,3))
        else:
            self.R = R
    
        # Default L is same shape as V. If L is a number, make flat map.
        if L is None:
            sx, sy, sz = V.shape
            self.L = np.zeros((sx,sy,sz))
        elif isinstance(L, int):
            sx, sy, sz = V.shape
            self.L = np.ones((sx,sy,sz))*L
        else:
            self.L = L
    
    def __call__(self, y):
        """ Returns a region given an origin, rotation vector, and layer.
            y[0:3]: Origin location
            y[3:6]: Universal rotation
            y[6]:   Universal delta applied to layer (L = L-min(L) + this) """
        new_region = Region(V = self.V, R = self.R, L=(self.L-np.min(self.L)+y[6]), x0 = y[0:3])
        new_region.rotate(y[3:6])
        return new_region
        
    def __str__(self):
        """ Prints this Blueprint's value tensor. """
        return str(self.V)
        
    def __repr__(self):
        """ Prints this Blueprint's ID """
        return "<Blueprint ID: {0}>".format(self.ID)
        
    def __hash__(self):
        """ Hashes the ID of this object. """
        return hash(self.ID)
        
    def __eq__(self, other):
        """ Checks that all tensors are equal """
        return np.all(self.V == other.V) and \
                np.all(self.R == other.R) and \
                np.all(self.L == other.L)
    
    # Pickling and saving
    # FIXME: Implement this
    def __getstate__(self):
        raise NotImplemented
        
    def __setstate__(self, state):
        raise NotImplemented