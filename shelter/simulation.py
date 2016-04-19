import numpy as np

def generate_light(W, d_l = 2):
    """ Creates a light vector """
    # FIXME: Light shouldn't go through walls
    # FIXME: Light should follow inverse square law
    L = np.zeros(W.shape)
    sources = np.where(W == torch)
    for y_i, x_i in np.vstack(sources).T:
        #y_i, x_i = l_i
        y_u, x_l, y_d, x_r = (y_i - d_l), (x_i - d_l), (y_i + d_l), (x_i + d_l)
        if y_u < 0: y_u = 0
        if x_l < 0: x_l = 0
        if y_d >= W.shape[0]: y_d = W.shape[0]
        if x_r >= W.shape[1]: x_r = W.shape[1]
        L[y_u:y_d,x_l:x_r] = np.ones((d_l*2,d_l*2))
        
    return L

def generate_particles(W, L, Np=10):
    """ Used to generate random particles in the world.
        Parameter: W - World Matrix. L - Light Matrix. Np - Number of particles.
        Returns: Particle Matrix """
    P = W.copy()    # Create a simulation copy of the world
    for i in np.arange(Np):
        # Only place particles outside light
        # FIXME: Faster solution?
        complete = False
        while not complete:
            x_i = np.random.randint(0,W.shape[0])
            y_i = np.random.randint(0,W.shape[0])
            complete = L[y_i,x_i] != 1      # We are done when the location given is not lit
        
        P[y_i,x_i] = parti  # Place a particle here
    
    return P
    
def run_simulation(P):
    """ Used to update the score based on particle location.
        Parameter: P - Particle Matrix.
        Return: Score Matrix """
    #FIXME: This would be faster in Cython

    # Create a vectorized solution
    run = np.vectorize(lambda x, u, r, l, d: parti if x != wall and parti in [x,l,r,u,d] else x)
    
    # Run untill there is no change
    S = P.copy()
    count, new_count = np.sum(S == parti), 0
    while count != new_count:
        # Replace count with new_count
        count = new_count
        
        # Create neighbor matricies
        i, j = S.shape
        padded = np.full((i+1,j+1),parti)
        padded[0:i,0:j] = S
    
        u = np.roll(padded,1,axis=0)[0:i,0:j]
        r = np.roll(padded,-1 ,axis=1)[0:i,0:j]
        l = np.roll(padded,1,axis=1)[0:i,0:j]
        d = np.roll(padded,-1 ,axis=0)[0:i,0:j]
    
        # Run on the latest matrix
        S = run(S,u,r,l,d)
        
        # Get a new count to compare
        new_count = np.sum(S == parti)
        
        # Log it
        print(count, new_count)
        
    return S
    
def score(W, S):
    # Generate count of unique items
    Nwalls = np.sum(W == wall)
    Nsludge = np.sum(S == parti)
    
    # The score is the "free-space" that remains
    # FIXME: This can be generalized
    return Nsludge + Nwalls
    
if __name__=="__main__":
    # Define shape labels
    space, wall, parti, torch = 1, 2, 3, 4
    
    # Create World
    Ny, Nx = 100, 100
    W = np.full((Ny,Nx), space)    # This is our world
    
    # Create a basic house
    H = [[wall,wall,wall,wall],
         [wall,torch,space,wall],
         [wall,torch,space,wall],
         [wall,wall,wall,wall]]
    H = np.array(H)
    
    # Place the house in the world
    W[Ny/2:Ny/2+H.shape[0], Nx/2:Nx/2+H.shape[1]] = H
    
    # Run the Simulation
    L = generate_light(W, d_l = 2)
    P = generate_particles(W, L, Np=10)
    S = run_simulation(P)
    print(score(W,S))
