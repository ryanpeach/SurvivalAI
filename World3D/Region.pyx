import pyximport; pyximport.install(pyimport = True)
from libcpp cimport bool

import numpy as np
cimport numpy as np

cdef dict LABEL_LOOKUP = {}
cdef dict COLOR_LOOKUP = {}

cdef np.ndarray toV3(np.ndarray V):
    cdef int dims = V.ndim
    if dims==2 and (V.shape[0] == 4 or V.shape[0] == 3) and V.shape[1] == 1:
        return np.array([V[0,0],V[1,0],V[2,0]], dtype=V.dtype)
    elif dims==1 and V.shape[0] == 4:
        return V[:3]
    elif dims==1 and V.shape[0] == 3:
        return V
    else:
        raise Exception("Input not the right shape.")
        
cdef np.ndarray to4x1(np.ndarray V, float fill_v = 0.):
    V = toV3(V)
    return np.array([[V[0]],[V[1]],[V[2]],[fill_v]], dtype="float")
    
cdef np.ndarray to3x1(np.ndarray V):
    V = toV3(V)
    return np.array([[V[0]],[V[1]],[V[2]]])
    
cdef np.ndarray unitV(np.ndarray V):
    V = toV3(V)
    return V / np.linalg.norm(V)
    
cdef np.ndarray rotM(float ax, float ay, float az):
    cdef float c1 = np.cos(ax), c2 = np.cos(ay), c3 = np.cos(az)
    cdef float s1 = np.sin(ax), s2 = np.sin(ay), s3 = np.sin(az)
    cdef np.ndarray R1 = np.array([[1,0,0,0]   ,[0,c1,s1,0],[0,s1,c1,0] ,[0,0,0,1]], dtype="float")
    cdef np.ndarray R2 = np.array([[c2,0,s2,0] ,[0,1,0,0]  ,[-s2,0,c2,0],[0,0,0,1]], dtype="float")
    cdef np.ndarray R3 = np.array([[c3,-s3,0,0],[s3,c3,0,0],[0,0,1,0]   ,[0,0,0,1]], dtype="float")
    return np.dot(np.dot(R1,R2),R3)
    
cdef np.ndarray transM(int dx, int dy, int dz):
    return np.array([[1,0,0,dx],[0,1,0,dy],[0,0,1,dz],[0,0,0,1]], dtype="int")

cdef np.ndarray forward(int a, int b, int c, np.ndarray M, np.ndarray C):
    cdef np.ndarray cV = to4x1(C,0)
    cdef np.ndarray V = to4x1(np.array([a,b,c]),1.) - cV
    cdef np.ndarray abcV = np.dot(M, cV)
    return toV3(abcV)
    
cdef np.ndarray inverse(int d, int e, int f, np.ndarray M, np.ndarray C):
    cdef np.ndarray cV = to4x1(C,0)
    cdef np.ndarray defV = to4x1(np.array([d,e,f]), 1.)
    cdef np.ndarray V = np.dot(np.inv(M), defV)
    cdef np.ndarray abcV = V + cV
    return toV3(abcV)
    
# Create Vectorized combination algorithm
cdef float layer_select(Region r0, Region r1, int x, int y, int z, np.ndarray M0, np.ndarray M1, float fill = 0.):
    cdef np.ndarray I0 = r0.xyz_ijk(x, y, z)
    cdef int i0 = I0[0], j0 = I0[1], k0 = I0[2]
    cdef int l0 = r0.layerM[i0,j0,k0]
    cdef bool v0 = r0.valid(i0,j0,k0)
    cdef np.ndarray I1 = r1.xyz_ijk(x, y, z)
    cdef int i1 = I1[0], j1 = I1[1], k1 = I1[2]
    cdef int l1 = r1.layerM[i1,j1,k1]
    cdef bool v1 = r1.valid(i1,j1,k1)
    if v0 and v1:
        if l1 > l0:
            return M1[i1,j1,k1]
        else:
            return M0[i0,j0,k0]
    elif v0:
        return M0[i0,j0,k0]
    elif v1:
        return M1[i1,j1,k1]
    else:
        return fill
        
cdef class Region:

    cdef np.ndarray spaceM, angleM, layerM
    cdef np.ndarray shape, center_ijk, center_uvw, center_xyz
    cdef np.ndarray U, X
    
    def __init__(self, np.ndarray spaceM, np.ndarray angleM, np.ndarray layerM = None, \
                  np.ndarray location_xyz = np.array([0,0,0]), \
                  np.ndarray rotation_angs = np.array([0,0,0]), \
                  np.ndarray rotation_ijk = None):
        # Set rules
        assert(spaceM.shape[0] == angleM.shape[0] and \
               spaceM.shape[1] == angleM.shape[1] and \
               spaceM.shape[2] == angleM.shape[2])

        # Set spacial data
        self.spaceM = np.array(spaceM, dtype="uint")
        self.angleM = np.array(angleM, dtype="float64")
        self.layerM = None
        if layerM.shape == angleM.shape:
            self.layerM = np.array(layerM, dtype="int")
        else:
            self.flatten(layerM)
            
        # Set info
        cdef np.npy_intp * shape = spaceM.shape  #FIXME: Check this works...
        cdef int di = shape[0], dj = shape[1], dk = shape[2]
        self.shape      = np.array([di,dj,dk])
        self.center_ijk = self.shape/2
        self.center_uvw = np.array([0,0,0])
    
        # Set Conversion Matricies
        self.U = np.eye(4)
        self.X = np.eye(4)
        
        # Move and rotate to initial points
        self.place(location_xyz)
        self.rotate(rotation_angs, rotation_ijk)
    
    cpdef rotate(self, np.ndarray rotation_ang, np.ndarray vector_ijk = None):
        if vector_ijk == None:
            vector_ijk = self.center_ijk
        vector_uvw = self.ijk_uvw(*vector_ijk)
        cdef int rx = rotation_ang[0], ry = rotation_ang[1], rz = rotation_ang[2]
        cdef int u = vector_uvw[0], v = vector_uvw[1], w = vector_uvw[2]
        cdef np.ndarray T1 = transM(u, v, w)                                    # Translate the the new point
        cdef np.ndarray R = rotM(rx, ry, rz)                                    # Rotate axes
        cdef np.ndarray V2 = np.dot(R, to4x1(-vector_uvw, 1))                   # Invert vector and rotate
        cdef np.ndarray vR = toV3(V2)                                           # Convert to a len==3 array
        cdef int vrx = vR[0], vry = vR[1], vrz = vR[2]
        cdef np.ndarray T2 = transM(vrx, vry, vrz)                              # Translate back to original point
        self.U = np.dot(T2, np.dot(R, np.dot(T1, self.U)))                      # Execute
        self.center_uvw = toV3(np.dot(self.U, to4x1(self.center_ijk)))          # Update Center
        assert(np.isclose(self.center_uvw, [0,0,0], rtol=0.4999, atol=0.0))     # center_uvw should always be close to 0.0
        
    cpdef move(self, np.ndarray vector_xyz):
        cdef int x = vector_xyz[0], y = vector_xyz[1], z = vector_xyz[2]
        cdef np.ndarray T = transM(x, y, z)
        self.X = np.dot(T, self.X)               # Translate to the new point
        self.center_xyz += np.array(vector_xyz)  # For ease of access
        
    cpdef place(self, np.ndarray location_xyz):
        cdef int x = location_xyz[0], y = location_xyz[1], z = location_xyz[2]
        cdef np.ndarray T = transM(x, y, z)
        self.X = T                                # Set X to the new point
        self.center_xyz = np.array(location_xyz)  # For ease of access
        
    cpdef flatten(self, int l = 0):
         """ Flattens the layer array to a single value. """
         self.layerM = np.full(self.shape, l, dtype="int")
    
    cpdef np.ndarray ijk_uvw(self, int i, int j, int k):
        return forward(i, j, k, self.U, self.center_ijk)
        
    cpdef np.ndarray uvw_ijk(self, int u, int v, int w):
        return inverse(u, v, w, self.U, self.center_ijk)
        
    cpdef np.ndarray uvw_xyz(self, int u, int v, int w):
        return forward(u, v, w, self.X, self.center_uvw)
        
    cpdef np.ndarray xyz_uvw(self, int x, int y, int z):
        return inverse(x, y, z, self.X, self.center_uvw)
        
    cpdef np.ndarray ijk_xyz(self, int i, int j, int k):
        cdef np.ndarray U = self.ijk_uvw(i, j, k)
        cdef int u = U[0], v = U[1], w = U[2]
        return self.uvw_xyz(u, v, w)
    
    cpdef np.ndarray xyz_ijk(self, int x, int y, int z):
        cdef np.ndarray I = self.xyz_uvw(x, y, z)
        cdef int i = I[0], j = I[1], k = I[2]
        return self.uvw_ijk(i, j, k)

    cpdef tuple getCorners(self):
        cdef int si = self.shape[0], sj = self.shape[1], sk = self.shape[2]
        cdef np.ndarray I = np.array([0.,si,si,0.,0.,si,si,0.])
        cdef np.ndarray J = np.array([sj,sj,sj,sj,0.,0.,0.,0.])
        cdef np.ndarray K = np.array([sk,sk,0.,0.,sk,sk,0.,0.])
        cdef tuple out = np.vectorize(self.ijk_xyz)(I, J, K)
        return out
        
    cpdef valid(self, int i, int j, int k):
        cdef int si = self.shape[0], sj = self.shape[1], sk = self.shape[2]
        return i >= 0 and j >= 0 and k >= 0 and i < si and j < sj and k < sk
    
    cpdef np.ndarray getAngle(self):
        cdef np.ndarray Vu = np.array([1.,0.,0.])
        cdef np.ndarray Vv = np.array([0.,1.,0.])
        cdef np.ndarray Vw = np.array([0.,0.,1.])
        cdef np.ndarray au = np.dot(unitV(np.dot(self.U, Vu)), Vu)
        cdef np.ndarray av = np.dot(unitV(np.dot(self.U, Vv)), Vv)
        cdef np.ndarray aw = np.dot(unitV(np.dot(self.U, Vw)), Vw)
        return np.array([au, av, aw])
        
    cpdef np.ndarray getAngleM(self):
        cdef np.ndarray angles = self.getAngle().reshape((1,1,1,3))
        return np.copy(self.angleM) + angles
        
    cpdef Region add(self, Region other):
        # Find the new bounds of the new arrays and the new location_xyz
        cdef np.ndarray cX0 = self.getCorners()
        cdef np.ndarray cX1 = other.getCorners()
        cdef np.ndarray cX = np.concatenate(cX0[0],cX1[0]), cY = np.concatenate(cX0[1],cX1[1]), cZ = np.concatenate(cX0[2],cX1[2])
        cdef int x0 = np.min(cX), y0 = np.min(cY), z0 = np.min(cZ)
        cdef int x1 = np.max(cX), y1 = np.max(cY), z1 = np.max(cZ)
        cdef np.ndarray location_xyz = self.center_xyz/2. + other.center_xyz/2.
        cdef int di =  self.shape[0], dj =  self.shape[1], dk = self.shape[2]
        
        cdef np.ndarray spaceM = np.empty((di, dj, dk), dtype="uint")
        cdef np.ndarray layerM = np.empty((di, dj, dk), dtype="int")
        cdef np.ndarray angleM = np.empty((di, dj, dk, 3), dtype="float64")
        
        cdef np.ndarray A0 = self.getAngleM(), A1 = other.getAngleM()
        cdef int x, y, z, i, j, k
        for i in 0 <= i < di:
            x = i + x0
            for j in 0 <= j < dj:
                y = j + y0
                for k in 0 <= k < dk:
                    z = k + z0
                    spaceM[i,j,k]   = layer_select(self, other, x, y, z, self.spaceM, other.spaceM)
                    layerM[i,j,k]   = layer_select(self, other, x, y, z, self.layerM, other.layerM)
                    angleM[i,j,k,:] = layer_select(self, other, x, y, z, A0, A1, 0.)
        
        return Region(spaceM, angleM, layerM, location_xyz)
    
    cpdef Region copy(self):
        return Region(V = self.V.copy(), R = self.R.copy(), L = self.L.copy(), x0 = tuple(self.x0))
    
    def draw(self, coord_ijk = True):
        # Source http://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        out, names = [], []
        for blkid in np.unique(self.shapeM):
            I, J, K = np.where(self.shapeM == blkid)
            if not coord_ijk:
                out.append(ax.scatter(I, J, K, zdir='z', s=20, label = LABEL_LOOKUP[blkid], depthshade=True))
                names.append(LABEL_LOOKUP[blkid])
            
        ax.legend(handles = out, labels = names)
        plt.show()
        
    
        
# class Region(Primitive):
#     """ A 3D space with a location, shape, value tensor, rotation tensor, and layer tensor.
#         Mutable, can be added, and contains a graph of all it's parents/children. """
#     SIZE = 7
#     def __init__(self, V, R = None, L = None, x0 = (0, 0, 0)):
#         super(Region, self).__init__()
        
#         # Set required properties with types
#         self.V = np.array(V, dtype='int')
#         self.dx = np.array(V.shape, dtype='int')
        
#         # Set default properties
#         # Default R is same shape as V with an extra dimension size 3
#         if R is None:
#             sx, sy, sz = V.shape
#             self.R = np.zeros((sx,sy,sz,3), dtype='int')
#         else:
#             self.R = R
        
#         # Default L is the same shape as V, either filled with a single value L, or 0
#         if L is None:
#             if isinstance(L, int):
#                 self.L = np.full(V.shape, fill_value = L, dtype='int')
#             elif isinstance(L, float):
#                 raise TypeError("L may not be a float")
#             else:
#                 self.L = np.zeros(V.shape, dtype='int')
#         else:
#             self.L = L
        
#         # Set other default properties
#         self.Vindex = {}
#         self.x0 = x0
    
#     # Accessor Methods
#     def __call__(self, x):
#         new_loc = np.array(x[:3], dtype='int')
#         new_L = self.L-np.min(self.L)+x[6]
#         out = Region(V = self.V, R = self.R, L = new_L, x0 = new_loc)
#         out = out.rotate(x[3:6])                                                # FIXME: Creates yet another Region, waste of memory, otherwise complicated
#         return out
    
#     def move(self, dx):
#         """ Move relative to current x0. """
#         new_loc = np.array(self.x0, dtype='int')+np.array(dx, dtype='int')
#         return Region(V = self.V, R = self.R, L = self.L, x0 = new_loc)
        
#     def place(self, x):
#         """ Place absolute location. """
#         new_loc = np.array(x, dtype='int')
#         return Region(V = self.V, R = self.R, L = self.L, x0 = new_loc)
        
#     def flatten(self, l = 0):
#         """ Flattens the layer array to a single value. """
#         new_L = np.ones(self.L.shape, dtype='int') * int(l)
#         return Region(V = self.V, R = self.R, L = new_L, x0 = self.x0)
        
#     def __getitem__(self, v):
#         """ Lazy indexing of V """
#         if self.Vindex == None:
#             self.Vindex = {}
#         if v not in self.Vindex:
#             self.Vindex[v] = np.where(self.V == v)
#         return self.Vindex[v]
    
#     def __str__(self):
#         return str(self.V)
        
#     # Modifier Methods
#     def __add__(self, other):
#         """ Merges two regions together. Favors "other" if layer levels are the same.
#             Returns new Region. """
#         # Import values from objects
#         x0, dx0, R0, L0, V0 = self.x0, self.dx, self.R, self.L, self.V          # Get both worlds with their coordinates and values
#         x1, dx1, R1, L1, V1 = other.x0, other.dx, other.R, other.L, other.V     # --
        
#         # Retrieve Relevant Data
#         x0, x1 = np.array([x0, x0+dx0-1]).T, np.array([x1, x1+dx1-1]).T             # Converting to legacy format for code compatability
#         min_x, min_y, min_z = min(x0[0,0], x1[0,0]), min(x0[1,0], x1[1,0]), min([x0[2,0], x1[2,0]])  # Get the bounds in each coordinate
#         max_x, max_y, max_z = max(x0[0,1], x1[0,1]), max(x0[1,1], x1[1,1]), max([x0[2,1], x1[2,1]])  # --
#         dx, dy, dz = max_x-min_x+1, max_y-min_y+1, max_z-min_z+1
        
#         # Transform x0 and x1 to world coordinates
#         origin = np.array([min_x, min_y, min_z]).T
#         x0, x1 = x0 - origin[:,np.newaxis], x1 - origin[:,np.newaxis]

#         # Transform v, l, and r to world coordinates
#         V, L, R = np.zeros((dx,dy,dz)), np.zeros((dx,dy,dz))-1, np.zeros((dx,dy,dz,3))                # Initialize variables
#         vt0, vt1, lt0, lt1, rt0, rt1 = V.copy(), V.copy(), L.copy(), L.copy(), R.copy(), R.copy()     # Copy to temps
#         vt0[x0[0,0]:x0[0,1]+1, x0[1,0]:x0[1,1]+1, x0[2,0]:x0[2,1]+1]    = V0                          # Insert region 0 and 1 into slices of superregion
#         vt1[x1[0,0]:x1[0,1]+1, x1[1,0]:x1[1,1]+1, x1[2,0]:x1[2,1]+1]    = V1
#         lt0[x0[0,0]:x0[0,1]+1, x0[1,0]:x0[1,1]+1, x0[2,0]:x0[2,1]+1]    = L0
#         lt1[x1[0,0]:x1[0,1]+1, x1[1,0]:x1[1,1]+1, x1[2,0]:x1[2,1]+1]    = L1
#         rt0[x0[0,0]:x0[0,1]+1, x0[1,0]:x0[1,1]+1, x0[2,0]:x0[2,1]+1, :] = R0
#         rt1[x1[0,0]:x1[0,1]+1, x1[1,0]:x1[1,1]+1, x1[2,0]:x1[2,1]+1, :] = R1
#         V0, V1, L0, L1, R0, R1 = vt0, vt1, lt0, lt1, rt0, rt1                       # Rename to original names
        
#         # Create the overlap functions vectorized
#         overlapV = np.vectorize(lambda v0, l0, v1, l1: v0 if l0 > l1 else v1)       # Define vectorized overlap function to favor v1 unless l0 is greater than l1
#         overlapL = np.vectorize(lambda l0, l1: l0 if l0 > l1 else l1)               # Do this for both the layer and the values
        
#         # Merge the layers
#         dim3 = lambda l: np.repeat(l[:,:,:,np.newaxis], 3, axis=3)
#         V = overlapV(V0, L0, V1, L1)                                                # Overlap worlds based on maximum layer
#         R = overlapV(R0, dim3(L0), R1, dim3(L1))                                    # Overlap world rotations based on maximum layer
#         L = overlapL(L0, L1)                                                        # Update the layer information
        
#         # Create new region
#         new_region = Region(V = V, R = R, L = L, x0 = (min_x, min_y, min_z))  # Return region
        
#         return new_region
    
#     def rotate(self, r):
#         """ Rotate the region increments of 90 degrees about each axis.
#             Params: r: A size 3 vector containing # 90 degree increments to rotate about each axis of rotation.
#             Returns new Region. """
            
#         # Rotate around 1st axis
#         V = nd_rot90(self.V, int(r[0]*90), axes=(0,1))
#         R = nd_rot90(self.R, int(r[0]*90), axes=(0,1))
#         L = nd_rot90(self.L, int(r[0]*90), axes=(0,1))
        
#         # Rotate around 2nd axis
#         V = nd_rot90(V, int(r[1]*90), axes=(1,2))
#         R = nd_rot90(R, int(r[1]*90), axes=(1,2))
#         L = nd_rot90(L, int(r[1]*90), axes=(1,2))
        
#         # Rotate around 3rd axis
#         V = nd_rot90(V, int(r[2]*90), axes=(2,0))
#         R = nd_rot90(R, int(r[2]*90), axes=(2,0))
#         L = nd_rot90(L, int(r[2]*90), axes=(2,0))
        
#         return Region(V = V, R = R, L = L, x0 = self.x0)                        # Return new Region, same origin
        
#     # Storing and immutability
#     # Only dx, and self.V are immutable
#     def __setattr__(self, name, val):
#         """ Controls value saving, immutables and typing. """
#         if name == 'x0':
#             val = np.array(val)
#             if val.shape != (3,):
#                 raise ValueError("{0} must be numpy array shape (3,). (Actual: {1})".format(name, val.shape))
#         if name in ('children', 'PARENTS') and val is not None:
#             val = list(val)
#             for i in range(len(val)):
#                 if isinstance(val[i], Region):
#                     val[i] = val[i].ID
#             if not all([isinstance(val[i], type(self.ID)) for i in range(len(val))]):
#                 raise TypeError("{0} must be iterable containing type Region or UUID".format(name))
#             val = tuple(val)
#         super(Region, self).__setattr__(name, val)
        
#     def draw(self):
#         # Source http://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
#         import matplotlib.pyplot as plt
#         from mpl_toolkits.mplot3d import Axes3D
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
        
#         out, names = [], []
#         for v in np.unique(self.V):
#             X, Y, Z = np.where(self.V == v)
#             out.append(ax.scatter(X, Y, Z, zdir='z', s=20, label = LOOKUP[v], depthshade=True))
#             names.append(LOOKUP[v])
            
#         ax.legend(handles = out, labels = names)
#         plt.show()
        
#     def __copy__(self):
#         return Region(V = self.V.copy(), R = self.R.copy(), L = self.L.copy(), x0 = tuple(self.x0))