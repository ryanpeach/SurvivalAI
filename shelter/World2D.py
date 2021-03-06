import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pdb

# Define shape labels
KEY = {'space': 0, 'wall': 1, 'parti': 4, 'torch': 2, 'door': 3, 'bedrock': 5}
COLORS = {'space': 'white', 'wall': 'brown', 'parti': 'red', 'torch': 'blue', 'door': 'orange', 'bedrock': 'black'}

not_removable = [KEY['space'], KEY['bedrock']]
not_passable  = [KEY['wall'] , KEY['bedrock']]
enemy_not_passable = [KEY['wall'], KEY['door'], KEY['bedrock']]

def draw(world, name='World', path='./log/'):
    plt.figure()
    legend_colors = []
    for k, v in KEY.items():
        c = COLORS[k]
        iy, ix = np.where(world == v)
        if v in enemy_not_passable:
            r = 10
        else:
            r = 1
        plt.scatter(ix, iy, s = r, color=c, label=k)
        
    iy, ix = np.where(world == -1)
    plt.scatter(ix, iy, s = 100, color='green', label='player')
    
    plt.title(name)
    plt.xlim([-1,world.shape[1]])
    plt.ylim([-1,world.shape[0]])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(path+name+'.png')
    plt.close()

def random_world(Nx, Ny, original = None, items = [KEY['wall']], Ni = 10):
    if original == None: W = np.full((Ny,Nx), KEY['space'], dtype='float')
    else: W = original.copy()
    for n in range(Ni):
        i = items[np.random.randint(0, len(items))]
        x, y = np.random.randint(0,Nx,size=1), np.random.randint(0,Ny,size=1)
        W[y,x] = i
    return W

def generate_light(W, d_l = 2):
    """ Creates a light vector """
    # FIXME: Light shouldn't go through walls
    # FIXME: Light should follow inverse square law
    L = np.zeros(W.shape)
    sources = np.where(W == KEY['torch'])
    for y_i, x_i in np.vstack(sources).T:
        #y_i, x_i = l_i
        y_u, x_l, y_d, x_r = (y_i - d_l), (x_i - d_l), (y_i + d_l), (x_i + d_l)
        if y_u < 0: y_u = 0
        if x_l < 0: x_l = 0
        if y_d >= W.shape[0]: y_d = W.shape[0]
        if x_r >= W.shape[1]: x_r = W.shape[1]
        L[y_u:y_d,x_l:x_r] = np.ones((abs(y_u-y_d),abs(x_l-x_r)))               # Create a light array at distance d_l from center of torch
        
    return L

def generate_particles(W, L, p = KEY['parti']):
    """ Used to generate random particles in the world.
        Parameter: W - World Matrix. L - Light Matrix. Np - Number of particles.
        Returns: Particle Matrix """
    gen = np.vectorize(lambda w, l: p if l != 1 and w not in enemy_not_passable else w)
    P = gen(W,L)

    return P
    
def run_simulation(P, p = KEY['parti'], impassable = enemy_not_passable, fill = KEY['parti']):
    """ Used to update the score based on particle location.
        Parameter: P - Particle Matrix.
        Return: Score Matrix """
    #FIXME: This would be faster in Cython

    # Create a vectorized solution
    run = np.vectorize(lambda x, u, r, l, d: p if x not in impassable \
                                                   and p in [x,l,r,u,d] else x)
    
    # Run untill there is no change
    S = P.copy()
    count, new_count = np.sum(S == p), -1
    while count != new_count:
        # Replace count with new_count
        count = new_count
        
        # Create neighbor matricies
        i, j = S.shape
        padded = np.full((i+1,j+1), fill_value=fill)
        padded[0:i,0:j] = S
    
        u = np.roll(padded,1,axis=0)[0:i,0:j]
        r = np.roll(padded,-1 ,axis=1)[0:i,0:j]
        l = np.roll(padded,1,axis=1)[0:i,0:j]
        d = np.roll(padded,-1 ,axis=0)[0:i,0:j]
    
        # Run on the latest matrix
        S = run(S,u,r,l,d)
        
        # Get a new count to compare
        new_count = np.sum(S == p)
        
        # Log it
        #print(count, new_count)
    
    return S
    
def scoreWorld(S, C, safety_weight = 1000, freedom_weight = 1):
    # Generate count of particles and impassables
    # The score is the "free-space" that remains
    # FIXME: This can be sped up with Cython
    # Count enemy score
    scored = enemy_not_passable + [KEY['parti']]
    S = np.vectorize(lambda x: x in scored)(S)
        
    # Count character score
    scored = [-1]
    C = np.vectorize(lambda x: x in scored)(C)
        
    safe = np.logical_and(C,np.logical_not(S))
    return safety_weight*np.sum(safe)+freedom_weight*np.sum(C)

def simple_score(W, safety_weight = 1000, freedom_weight = 1):
    L = generate_light(W, d_l = 2)
    P = generate_particles(W, L, p = KEY['parti'])
    S = run_simulation(P, p = KEY['parti'], impassable = enemy_not_passable, fill = KEY['parti'])
    C = run_simulation(W, p = -1, impassable = not_passable, fill = -1)
    return scoreWorld(S,C,safety_weight = safety_weight, freedom_weight = freedom_weight)
        
# Define module tests
if __name__=="__main__":
    # Create World
    Ny, Nx = 10, 10
    W = np.full((Ny,Nx), KEY['space'])    # This is our world
    
    # Create a basic house
    H = [[KEY['wall'],KEY['wall'], KEY['wall'], KEY['wall']],
         [KEY['door'],KEY['space'],KEY['space'],KEY['wall']],
         [KEY['wall'],KEY['torch'],KEY['space'],KEY['wall']],
         [KEY['wall'],KEY['wall'], KEY['wall'], KEY['wall']]]
    H = np.array(H)
    
    # Place the house in the world
    W[Ny/2:Ny/2+H.shape[0], Nx/2:Nx/2+H.shape[1]] = H
    
    # Run the Simulation
    #L = generate_light(W, d_l = 2)
    #P = generate_particles(W, L)
    #S = run_simulation(P)
    #print(scoreWorld(W,S))
    
    print(simple_score(W))
