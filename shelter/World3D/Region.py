import numpy as np

class Region(object):
    """ A 3D space with a location, shape, value tensor, rotation tensor, and layer tensor.
        Mutable, can be added, and contains a graph of all it's parents/children. """
        
    def __init__(self, V, R = None, L = None, x0 = (0, 0, 0)):
        # Set immutable properties
        self.V = np.array(V)
        self.dx = np.array(V.shape)
        self.ID = uuid4()
        
        # Set mutable properties
        # Default R is same shape as V with an extra dimension size 3
        if R is None:
            sx, sy, sz = V.shape
            self.R = np.zeros((sx,sy,sz,3))
        else:
            self.R = R
        
        # Default L is the same shape as V, either filled with a single value L, or 0
        if L is None:
            if isinstance(L, (int, long, float)):
                self.L = np.full(V.shape, fill_value = L)
            else:
                self.L = np.zeros(V.shape)
        else:
            self.L = L
        
        # Set other default properties
        self.Vindex = {}
        self.x0 = x0
    
    # Accessor Methods
    def move(self, dx):
        """ Move relative to current x0. """
        self.x0 += np.array(dx)
        
    def place(self, x):
        """ Place absolute location. """
        self.x0 = x
        
    def flatten(self, l = 0.):
        """ Flattens the layer array to a single value. """
        self.L = np.ones(self.L.shape) * l
        
    def __getitem__(self, v):
        """ Lazy indexing of V """
        if self.Vindex == None:
            self.Vindex = {}
        if v not in self.Vindex:
            self.Vindex[v] = np.where(self.V == v)
        return self.Vindex[v]
    
    def get_parents(self, lookup):
        """ Returns node parents given a lookup table of ID's """
        out = []
        if self.PARENTS is not None:
            for p in self.PARENTS:
                out.append(lookup[p])
        return out
        
    def get_children(self, lookup):
        """ Returns node children given a lookup table of ID's """
        out = []
        for c in self.children:
            out.append(lookup[c])
        return out
        
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
        
    def __eq__(self, other):
        """ Checks that all immutable properties are equals """
        out = []
        for name in ('V','R','L'):
            out.append(np.all(self.__getattr___(name) == other.__getattr__(name)))
        return np.all(out)

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