import numpy as np

class Region(object):
    __slots__ = ['x0','x1','R','L','V']
    def __init__(self, x0, x1, R, L, V):
        assert(x0[0]<=x1[0] and x0[1]<=x1[1] and x0[2]<=x1[2])                  # x0 must be less than x1
        self.x0, self.x1 = np.array(x0), np.array(x1)                           # Set properties
        self.R, self.L, self.V = np.array(R), np.array(L), np.array(V)          # --
        
    def __add__(self, other):
        """ Warning: Favors other if layer levels are the same. """
        # Declarations
        x0, R0, L0, V0 = np.array([self.x0, self.x1]).T, self.R, self.L, self.V   # Get both worlds with their coordinates and values
        x1, R1, L1, V1 = np.array([other.x0, other.x1]).T, other.R, other.L, other.V 
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
        V0, V1, L0, L1, R0, R1 = vt0, vt1, lt0, lt1, rt0, rt1                                       # Rename to original names
        
        # Create the overlap functions vectorized
        overlapV = np.vectorize(lambda v0, l0, v1, l1: v0 if l0 > l1 else v1)       # Define vectorized overlap function to favor v1 unless l0 is greater than l1
        overlapL = np.vectorize(lambda l0, l1: l0 if l0 > l1 else l1)               # Do this for both the layer and the values
        
        # Merge the layers
        dim3 = lambda l: np.repeat(l[:,:,:,np.newaxis], 3, axis=3)
        V = overlapV(V0, L0, V1, L1)                                                # Overlap worlds based on maximum layer
        R = overlapV(R0, dim3(L0), R1, dim3(L1))                                    # Overlap world rotations based on maximum layer
        L = overlapL(L0, L1)                                                        # Update the layer information
        
        return Region([min_x, min_y, min_z], [max_x, max_y, max_z], R, L, V)     # Return region
    
    def translate(self, x):
        x0 = self.x0 + x
        x1 = self.x1 + x
        return Region(x0, x1, self.R.copy(), self.L.copy(), self.V.copy())
        
    def rotate(self, r):
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
        x1 = np.array(V.shape) + self.x0                                        # Update non-origin corner
        return Region(self.x0.copy(), x1, R, V, L)                              # Return new Region
        
    def flip(self, vert = True):
        R, V, L = self.R.copy(), self.L.copy(), self.V.copy()                   # Get a copy of all object info
        if vert:                                                                # If vert is selected
            R, V, L = np.flipud(R), np.flipud(V), np.flipud(L)                  # Flip each up and down
        else:                                                                   # Otherwise
            R, V, L = np.fliplr(R), np.fliplr(V), np.fliplr(L)                  # Flip each left and right
        x1 = np.array(V.shape) + self.x0                                        # Update non-origin corner
        return Region(self.x0.copy(), x1, R, V, L)                              # Return new Region
    
    def __str__(self):
        return str(self.V)
    
    def __hash__(self):
        return hash(str(self.x0)+str(self.x1)+str(self.R)+str(self.L)+str(self.V))
        
    def __eq__(self, other):
        x0_eq = np.all(self.x0 == other.x0)
        x1_eq = np.all(self.x1 == other.x1)
        R_eq  = np.all(self.R == other.R)
        L_eq  = np.all(self.L == other.L)
        V_eq  = np.all(self.V == other.V)
        return np.all([x0_eq, x1_eq, R_eq, L_eq, V_eq])
        
def block(block_id, x0 = (0,0,0), r = (0,0,0), l = 0.):
    """ Params: block_id: scalar; The search key for all block information.
                x0: vector (3,); Location vector.
                r : vector (3,); Rotation vector.
                l : scalar; The layer the block exists in (used in merging). """
    R = np.ones((1,1,1,3)) * r                                                  # Initialize the rotation tensor
    L = np.full((1,1,1), fill_value = float(l))                                 # Initialize the layer value tensor
    V = np.full((1,1,1), fill_value = float(block_id))                          # Initialize the block id tensor
    return Region(x0, x0, R, L, V)                                              # Return the region
    
def cube(block_id, x1, x0 = (0,0,0), r = (0,0,0), l = 0):
    """ Params: x0: vector (3,); Starting location vector.
                x1: vector (3,); Ending location vector.
                r : vector (3,); Rotation vector.
                l : scalar; The layer the cube exists in (used in merging). """
    dx, dy, dz = x1[0]-x0[0]+1, x1[1]-x0[1]+1, x1[2]-x0[2]+1
    R = np.ones((dx, dy, dz, 3)) * r                                            # Set the rotation matrix using r vector
    L = np.full((dx, dy, dz), fill_value = float(l))                            # Initialize layer with layer value
    V = np.full((dx, dy, dz), fill_value = float(block_id))                     # Initialize the value with block_id
    return Region(x0, x1, R, L, V)                                              # Return the region

      
if __name__ == "__main__":
    a = cube(1, [1,1,1])
    b = cube(2, [1,3,3], [0,1,1])
    c = block(3, [0,3,0])
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