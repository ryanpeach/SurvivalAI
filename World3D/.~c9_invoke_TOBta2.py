import numpy as np

class Blueprint(object):
    """ The immutable 3D container for any space. """
    __slots__ = ("ID","V","R","dX")
    
    def __init__(self, name, block_map, rotation_map = None):
        self.ID = name
        self.V  = np.array(block_map)
        self.dX = np.array(self.V.shape)
        if rotation_map is None:
            self.R = np.array(rotation_map)
        else:
            self.R = np.zeros(self.dX)
    
    # Accessors
    def getData(self):
        return self.V.copy(), self.R.copy()
    def __getitem__(self, v):
        return self.V[v], self.R[v]
    def draw(self):
        raise NotImplementedError
    
    # Representation
    def __hash__(self):
        return self.ID
    def __repr__(self):
        return self.ID

class Region(object):
    """ A 3D space with a location, shape, value tensor, rotation tensor, and layer tensor.
        Mutable, can be added, and contains a graph of all it's parents/children. """
        
    def __init__(self, Bp, L = None, X0 = (0, 0, 0)):
        # Set Properties
        self.V, self.R = Bp.getData()
        self.dX = np.array(Bp.dX)
        self.X0 = np.array(X0)
        self.Vindex = None
        
        # Default L is the same shape as V, either filled with a single value L, or 0
        if L is None:
            if isinstance(L, (int, long, float)):
                self.L = np.full(self.V.shape, fill_value = L)
            else:
                self.L = np.zeros(self.V.shape)
        else:
            self.L = np.array(L)
        

    # Simple Modifier Methods
    def move(self, dX):
        """ Move relative to current x0. """
        self.X0 += np.array(dX)
        
    def place(self, X):
        """ Place absolute location. """
        self.X0 = X
        
    def flatten(self, l = 0.):
        """ Flattens the layer array to a single value. """
        self.L = np.ones(self.L.shape) * l
        
    # Combination Methods
    def __add__(self, other):
        """ Merges two regions together. Favors "other" if layer levels are the same.
            Returns new Region. """
        # Import values from objects
        x0, dx0, R0, L0, V0 = self.X0, self.dX, self.R, self.L, self.V          # Get both worlds with their coordinates and values
        x1, dx1, R1, L1, V1 = other.X0, other.dX, other.R, other.L, other.V     # --
        
        # Retrieve Relevant Data
        x0, x1 = np.array([x0, x0+dx0-1]).T, np.array([x1, x1+dx1-1]).T                              # Converting to legacy format for code compatability
        min_x, min_y, min_z = min(x0[0,0], x1[0,0]), min(x0[1,0], x1[1,0]), min([x0[2,0], x1[2,0]])  # Get the bounds in each coordinate
        max_x, max_y, max_z = max(x0[0,1], x1[0,1]), max(x0[1,1], x1[1,1]), max([x0[2,1], x1[2,1]])  # --
        dx, dy, dz = max_x-min_x+1, max_y-min_y+1, max_z-min_z+1
        
        # Transform x0 and x1 to world coordinates
        origin = np.array([[min_x], [min_y], [min_z]])
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
    
    # Complex Modifiers
    def rotate(self, r):
        """ Rotate the region increments of 90 degrees about each axis.
            Params: r: A size 3 vector containing # 90 degree increments to rotate about each axis of rotation.
            Returns new Region.
            FIXME: This method is untested and likely does not effect the right axes. """
        R, V, L = self.R.copy(), self.L.copy(), self.V.copy()                   # Get a copy of all object info
        R, V, L = np.rot90(R, r[0]), np.rot90(V, r[0]), np.rot90(L, r[0])       # Rotate each by first value
        R, V, L = np.swapaxes(R,0,2), np.swapaxes(V,0,2), np.swapaxes(L,0,2)    # Roll axes
        R, V, L = np.swapaxes(R,1,2), np.swapaxes(V,1,2), np.swapaxes(L,1,2)    # --
        R, V, L = np.rot90(R, r[1]), np.rot90(V, r[1]), np.rot90(L, r[1])       # Rotate each by second value
        R, V, L = np.swapaxes(R,0,2), np.swapaxes(V,0,2), np.swapaxes(L,0,2)    # Roll axes
        R, V, L = np.swapaxes(R,1,2), np.swapaxes(V,1,2), np.swapaxes(L,1,2)    # --
        R, V, L = np.rot90(R, r[2]), np.rot90(V, r[2]), np.rot90(L, r[2])       # Rotate each by third value
        R, V, L = np.swapaxes(R,0,2), np.swapaxes(V,0,2), np.swapaxes(L,0,2)    # Roll axes back to original
        R, V, L = np.swapaxes(R,1,2), np.swapaxes(V,1,2), np.swapaxes(L,1,2)    # --
        
        return Region(V = V, x0 = self.x0, R = R, L = L)                        # Return new Region
        
    def flip(self, r):
        """ Flips the array about one of the axes in r.
            Params: r is a size 3 iterable of type bool.
            Returns a new Region.
            FIXME: This method is untested and likely does not effect the right axes. """
        assert(not ((r[0] and r[1]) or (r[1] and r[2]) or (r[2] and r[0])), "Only one index may be true")
        R, V, L = self.R.copy(), self.L.copy(), self.V.copy()                   # Get a copy of all object info
        if r[0]:                                                                # If vert is selected
            R, V, L = np.flipud(R), np.flipud(V), np.flipud(L)                  # Flip each up and down
        elif r[1]:                                                              # Otherwise
            R, V, L = np.fliplr(R), np.fliplr(V), np.fliplr(L)                  # Flip each left and right
        elif r[2]:
            R, V, L = np.swapaxes(R,0,1), np.swapaxes(V,0,1), np.swapaxes(L,0,1)# Roll axes
            R, V, L = np.fliplr(R), np.fliplr(V), np.fliplr(L)                  # Flip each left and right
            R, V, L = np.swapaxes(R,1,0), np.swapaxes(V,1,0), np.swapaxes(L,1,0)# --
        return Region(V = V, x0 = self.x0, R = R, L = L)                        # Return new Region
    
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
        
    def __eq__(self, other):
        """ Checks that all immutable properties are equals """
        out = []
        for name in ('V','R','L'):
            out.append(np.all(self.__getattr___(name) == other.__getattr__(name)))
        return np.all(out)
        
    def __str__(self):
        return str(self.V)