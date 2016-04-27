import numpy as np

class PaternTrainer(object):
    def __init__(self, examples):
        self.EX = examples  # Create list of examples
        
class Pattern(object):
    # Define Globals
    GREATER_INDEX, LESSER_INDEX, EQUALS_INDEX = 0, 1, 2
    
    def __init__(self, block_options):
        """ Block options is a 2D array that lists blocks by their number in axis 0, and lists options for that block in axis 1 """
        # Create blocks list
        self.blocks, self.Nobjs = np.array(block_options), len(block_options)
        
        # The location poly in 4D space. shape = (#obj, [x,y,z,l], [x,*poly])
        # The size poly in 3D space. shape = (#obj, [dx,dy,dz], [x,*poly])
        # The rotation poly in 3D space. shape = (#obj, [rx,ry,rz], [x,*poly])
        # Combine into self._X
        self._X  = np.zeros([self.Nobjs, 10, 3])
        
        # Similarly, there are constraints
        self.GREATER = -np.ones([self.Nobjs, 10, 3])
        self.LESSER  =  np.ones([self.Nobjs, 10, 3])
        self.EQUALS  =  np.ones([self.Nobjs, 10, 3])
        
        # And we may select any of them we wish by signaling True in axis 3, following:
        # shape = (#obj, dim, poly, [GREATER, LESSER, EQUALS])
        self.SEL     =  np.full([self.Nobjs, 10, 3, 3], fill_value=False)
        
    def matrix_constraint(value, obj_n, dim_n, poly_n, sel_index, activate = True):
        """ Can be used to set many constraints at once, given a array value. """
        self.GREATER[obj_n, dim_n, poly_n] = value[:,:,:,self.GREATER_INDEX]
        self.LESSER[obj_n, dim_n, poly_n]  = value[:,:,:,self.LESSER_INDEX]
        self.EQUALS[obj_n, dim_n, poly_n]  = value[:,:,:,self.EQUALS_INDEX]
        self.SEL[obj_n, dim_n, poly_n, sel_index]  = activate
    
    def constraint(value, obj_n, dim_n, poly_n, sel_index, activate = True):
        """ Used to set just one type of constraint at a time. """
        if sel_index == self.GREATER_INDEX:
            self.GREATER[obj_n, dim_n, poly_n] = value
        elif sel_index == self.LESSER_INDEX:
            self.LESSER[obj_n, dim_n, poly_n]  = value
        elif sel_index == self.EQUALS_INDEX:
            self.EQUALS[obj_n, dim_n, poly_n]  = value
        else:
            raise KeyError("Selection Index {0} not an option.".format(sel_index))
            
        self.SEL[obj_n, dim_n, poly_n, kind_index]  = activate
            
    def __copy__(self):
        out = Pattern(deepcopy(self.blocks))
        out._X = deepcopy(self._X)
        out.GREATER = self.GREATER
        out.LESSER  = self.LESSER
        out.EQUALS  = self.EQUALS
        out.SEL     = self.SEL
        out.WORLD   = self.WORLD
        return out
        
    @staticmethod
    def initialize_world(X):
        """ Initializes a world to contain all the options in X """
        min_x, max_x = np.min(X[:,0,0]), np.max(X[:,0,0])+np.max(X[:,4,0])      # The max is approximated by using the max starting position + the max size
        min_y, max_y = np.min(X[:,1,0]), np.max(X[:,1,0])+np.max(X[:,5,0])
        min_z, max_z = np.min(X[:,2,0]), np.max(X[:,2,0])+np.max(X[:,6,0])
        dx, dy, dz = max_x-min_x, max_y-min_y, max_z-min_z
        X0, Y0, Z0 = np.meshgrid(np.arange(min_x, max_x), np.arange(min_y, max_y), np.arange(min_z, max_z))
        L0, V0 = np.zeros([dx,dy,dz]), np.zeros([dx,dy,dz])
        return X0, Y0, Z0, L0, V0
        
    @staticmethod
    def generate_blocks(blocks, X)
        world = Pattern.initialize_world(X)
        for i in np.arange(len(blocks)):
            section = cube(blocks[i], X[i,:,:])
            world = Pattern.merge3D(world, section)
        return world
            
    @staticmethod
    def merge3D(world, section):
        X0, Y0, Z0, L0, V0 = world                                              # Get the world with it's coordinates and values
        X1, Y1, Z1, L1, V1 = section                                            # Get the section along with it's coordinates and values
        
        overlapV = np.vectorize(lambda v0, l0, v1, l1: v0 if l0 > l1 else v1)   # Define vectorized overlap function to favor v1 unless l0 is greater than l1
        overlapL = np.vectorize(lambda l0, l1: l0 if l0 > l1 else l1)           # Do this for both the layer and the values
        
        mask = (X0==X1, Y0==Y1, Z0==Z1)                                         # Get the map where the two vectors overlap
        V0mask, L0mask = V0[*mask], L0[*mask]                                   # Preprocess the world with the mask
        
        V0mask = overlapV(V0mask, L0mask, V1, L1)                               # Overlap section onto world in region x1, y1, z1 based on l0 and l1
        L0mask = overlapL(L0mask, L1)                                           # Update the layer information
        return X0, Y0, Z0, L0, V0                                               # Return world
        
    @staticmethod
    def cube(blockid, Xi):
        X0, Y0, Z0, L0, V0 = Pattern.initialize_world(np.array([Xi]))           # Initialize world, giving it a batch dimension for compatability
        mask = Xi[0,0]:Xi[0,0]+Xi[4,0],
                   Xi[1,0]:Xi[1,0]+Xi[5,0], Xi[2,0]:Xi[2,0]+Xi[6,0]             # Preprocess mask based on Xi parameters
        V0[*mask] = blockid                                                     # Set the value in each location to blockid
        L0[*mask] = Xi[3,0]                                                     # Set the layer in each location to the layer specified in Xi
        return X0, Y0, Z0, L0, V0                                               # Return the section of the world