# Python Imports
from copy import copy

# Library Imports
import numpy as np
from scipy.ndimage.interpolation import rotate as nd_rot90

# Local Imports
from World3D import Primitive

class Region(Primitive):
    """ A 3D space with a location, shape, value tensor, rotation tensor, and layer tensor.
        Mutable, can be added, and contains a graph of all it's parents/children. """
    SIZE = 7
    def __init__(self, V, R = None, L = None, x0 = (0, 0, 0)):
        super(Region, self).__init__()
        
        # Set required properties with types
        self.V = np.array(V, dtype='int')
        self.dx = np.array(V.shape, dtype='int')
        
        # Set default properties
        # Default R is same shape as V with an extra dimension size 3
        if R is None:
            sx, sy, sz = V.shape
            self.R = np.zeros((sx,sy,sz,3), dtype='int')
        else:
            self.R = R
        
        # Default L is the same shape as V, either filled with a single value L, or 0
        if L is None:
            if isinstance(L, int):
                self.L = np.full(V.shape, fill_value = L, dtype='int')
            elif isinstance(L, float):
                raise TypeError("L may not be a float")
            else:
                self.L = np.zeros(V.shape, dtype='int')
        else:
            self.L = L
        
        # Set other default properties
        self.Vindex = {}
        self.x0 = x0
    
    # Accessor Methods
    def __call__(self, x):
        new_loc = np.array(x[:3], dtype='int')
        new_L = self.L-np.min(self.L)+x[6]
        out = Region(V = self.V, R = self.R, L = new_L, x0 = new_loc)
        out = out.rotate(x[3:6])                                                # FIXME: Creates yet another Region, waste of memory, otherwise complicated
        return out
        
    def move(self, dx):
        """ Move relative to current x0. """
        new_loc = np.array(self.x0, dtype='int')+np.array(dx, dtype='int')
        return Region(V = self.V, R = self.R, L = self.L, x0 = new_loc)
        
    def place(self, x):
        """ Place absolute location. """
        new_loc = np.array(x, dtype='int')
        return Region(V = self.V, R = self.R, L = self.L, x0 = new_loc)
        
    def flatten(self, l = 0):
        """ Flattens the layer array to a single value. """
        new_L = np.ones(self.L.shape, dtype='int') * int(l)
        return Region(V = self.V, R = self.R, L = new_L, x0 = self.x0)
        
    def __getitem__(self, v):
        """ Lazy indexing of V """
        if self.Vindex == None:
            self.Vindex = {}
        if v not in self.Vindex:
            self.Vindex[v] = np.where(self.V == v)
        return self.Vindex[v]
    
    def __str__(self):
        return str(self.V)
        
    # Modifier Methods
    def __add__(self, other):
        """ Merges two regions together. Favors "other" if layer levels are the same.
            Returns new Region. """
        # Import values from objects
        x0, dx0, R0, L0, V0 = self.x0, self.dx, self.R, self.L, self.V          # Get both worlds with their coordinates and values
        x1, dx1, R1, L1, V1 = other.x0, other.dx, other.R, other.L, other.V     # --
        
        # Retrieve Relevant Data
        x0, x1 = np.array([x0, x0+dx0-1]).T, np.array([x1, x1+dx1-1]).T             # Converting to legacy format for code compatability
        min_x, min_y, min_z = min(x0[0,0], x1[0,0]), min(x0[1,0], x1[1,0]), min([x0[2,0], x1[2,0]])  # Get the bounds in each coordinate
        max_x, max_y, max_z = max(x0[0,1], x1[0,1]), max(x0[1,1], x1[1,1]), max([x0[2,1], x1[2,1]])  # --
        dx, dy, dz = max_x-min_x+1, max_y-min_y+1, max_z-min_z+1
        
        # Transform x0 and x1 to world coordinates
        origin = np.array([min_x, min_y, min_z]).T
        x0, x1 = x0 - origin[:,np.newaxis], x1 - origin[:,np.newaxis]

        # Transform v, l, and r to world coordinates
        V, L, R = np.zeros((dx,dy,dz)), np.zeros((dx,dy,dz))-1, np.zeros((dx,dy,dz,3))                # Initialize variables
        vt0, vt1, lt0, lt1, rt0, rt1 = V.copy(), V.copy(), L.copy(), L.copy(), R.copy(), R.copy()     # Copy to temps
        vt0[x0[0,0]:x0[0,1]+1, x0[1,0]:x0[1,1]+1, x0[2,0]:x0[2,1]+1]    = V0                          # Insert region 0 and 1 into slices of superregion
        vt1[x1[0,0]:x1[0,1]+1, x1[1,0]:x1[1,1]+1, x1[2,0]:x1[2,1]+1]    = V1
        lt0[x0[0,0]:x0[0,1]+1, x0[1,0]:x0[1,1]+1, x0[2,0]:x0[2,1]+1]    = L0
        lt1[x1[0,0]:x1[0,1]+1, x1[1,0]:x1[1,1]+1, x1[2,0]:x1[2,1]+1]    = L1
        rt0[x0[0,0]:x0[0,1]+1, x0[1,0]:x0[1,1]+1, x0[2,0]:x0[2,1]+1, :] = R0
        rt1[x1[0,0]:x1[0,1]+1, x1[1,0]:x1[1,1]+1, x1[2,0]:x1[2,1]+1, :] = R1
        V0, V1, L0, L1, R0, R1 = vt0, vt1, lt0, lt1, rt0, rt1                       # Rename to original names
        
        # Create the overlap functions vectorized
        overlapV = np.vectorize(lambda v0, l0, v1, l1: v0 if l0 > l1 else v1)       # Define vectorized overlap function to favor v1 unless l0 is greater than l1
        overlapL = np.vectorize(lambda l0, l1: l0 if l0 > l1 else l1)               # Do this for both the layer and the values
        
        # Merge the layers
        dim3 = lambda l: np.repeat(l[:,:,:,np.newaxis], 3, axis=3)
        V = overlapV(V0, L0, V1, L1)                                                # Overlap worlds based on maximum layer
        R = overlapV(R0, dim3(L0), R1, dim3(L1))                                    # Overlap world rotations based on maximum layer
        L = overlapL(L0, L1)                                                        # Update the layer information
        
        # Create new region
        new_region = Region(V = V, R = R, L = L, x0 = (min_x, min_y, min_z))  # Return region
        
        return new_region
    
    def rotate(self, r):
        """ Rotate the region increments of 90 degrees about each axis.
            Params: r: A size 3 vector containing # 90 degree increments to rotate about each axis of rotation.
            Returns new Region. """
            
        # Rotate around 1st axis
        V = nd_rot90(self.V, int(r[0]*90), axes=(0,1))
        R = nd_rot90(self.R, int(r[0]*90), axes=(0,1))
        L = nd_rot90(self.L, int(r[0]*90), axes=(0,1))
        
        # Rotate around 2nd axis
        V = nd_rot90(V, int(r[1]*90), axes=(1,2))
        R = nd_rot90(R, int(r[1]*90), axes=(1,2))
        L = nd_rot90(L, int(r[1]*90), axes=(1,2))
        
        # Rotate around 3rd axis
        V = nd_rot90(V, int(r[2]*90), axes=(2,0))
        R = nd_rot90(R, int(r[2]*90), axes=(2,0))
        L = nd_rot90(L, int(r[2]*90), axes=(2,0))
        
        return Region(V = V, R = R, L = L, x0 = self.x0)                        # Return new Region, same origin
        
    # Storing and immutability
    # Only dx, and self.V are immutable
    def __setattr__(self, name, val):
        """ Controls value saving, immutables and typing. """
        if name == 'x0':
            val = np.array(val)
            if val.shape != (3,):
                raise ValueError("{0} must be numpy array shape (3,). (Actual: {1})".format(name, val.shape))
        if name in ('children', 'PARENTS') and val is not None:
            val = list(val)
            for i in range(len(val)):
                if isinstance(val[i], Region):
                    val[i] = val[i].ID
            if not all([isinstance(val[i], type(self.ID)) for i in range(len(val))]):
                raise TypeError("{0} must be iterable containing type Region or UUID".format(name))
            val = tuple(val)
        super(Region, self).__setattr__(name, val)
        
    def draw(self):
        # Source http://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        out, names = [], []
        for v in np.unique(self.V):
            X, Y, Z = np.where(self.V == v)
            out.append(ax.scatter(X, Y, Z, zdir='z', s=20, label = LOOKUP[v], depthshade=True))
            names.append(LOOKUP[v])
            
        ax.legend(handles = out, labels = names)
        plt.show()
        
    def __copy__(self):
        return Region(V = self.V.copy(), R = self.R.copy(), L = self.L.copy(), x0 = tuple(self.x0))