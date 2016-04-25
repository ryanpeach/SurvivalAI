from World2D import *
from np.random import choice, randint
from uuid import uuid4 as uuid
import pandas as pd

GENES = [KEY['wall'], KEY['torch'], KEY['door'], KEY['space']]
GENES_FREQ = [1,1,1,1]
GENES_FREQ = list(np.array(GENES_FREQ) / np.sum(GENES_FREQ))    # Normalize frequency

def combine(A, B):
    # Check type and get variables
    assert(A.shape == B.shape)
    Ny, Nx = A.shape
    Ns = ny*nx / 2
    
    # Copy A and B to prevent overriding data
    A, B = A.copy(), B.copy()

    # Generate coordinate array to sample
    X, Y = np.meshgrid(np.arange(Ny),np.arange(Nx))
    COORD = np.dstack((X,Y))
    COORD = np.reshape(COORD, [Ny*Nx,2])
    
    # Sample coordinates for A
    Ay = choice(COORD[:,0], size = [Ns], replace = False)
    Ax = choice(COORD[:,1], size = [Ns], replace = False)
    
    # Get the inverse of the sample for B
    By = np.setdiff1d(All[:,0], Ay)
    Bx = np.setdiff1d(All[:,1], Ax)
    
    # Create an output array and set values
    out = np.zeros((Ny,Nx))
    out[By,Bx] = B[By,Bx]
    out[Ay,Ax] = A[Ay,Ax]
    
    return out
    
def mutate(A, rate = 1):
    """ Mutates a world based on the GENES list.
        Simply replaces rates # of coordinates with random sampling from GENES """
    # Copy A to prevent overriding data, get its shape
    A = A.copy()
    Ny, Nx = A.shape
    
    # Get a selection of genes to insert into A
    sel = choice(GENES, size = rate, p = GENES_FREQ, replace = True)
    
    # Generate coordinate array to sample
    X, Y = np.meshgrid(np.arange(Ny),np.arange(Nx))
    COORD = np.dstack((X,Y))
    COORD = np.reshape(COORD, [Ny*Nx,2])
    
    # Sample coordinates for A
    Ay = choice(COORD[:,0], size = rate, replace = False)
    Ax = choice(COORD[:,1], size = rate, replace = False)
    
    # Place genes into coordinates
    A[Ay,Ax] = sel
    
    return A
    
def resample(gen, scores, percent_eliminate = .5, mutation_rate = 5):
    """ Breeds generation together based on their scores. Returns new generation. """
    Ns = len(gen)*(1.-percent_eliminate)  # Get the sample size
    P = np.array(scores) / np.sum(scores) # normalize scores to get probabilities
    
    # Sample the generation based on score, higher the more probable to remain
    A = np.choice(breed, size = Ns/2, replace = True, p = P)
    B = np.choice(breed, size = Ns/2, replace = True, p = P)
    
    # Generate the next generation
    new_gen, new_scores = [], []
    for a, b in zip(A,B):
        # Generate new world via gene combination and mutation
        new_world = combine(a,b)
        new_world = mutate(new_world, rate = mutation_rate)
        
        # Score the new world
        new_score = simple_score(new_world, safety_weight = Safety, freedom_weight = Freedom)
        
        # Append it to the new generation
        new_gen.append(new_world)
        new_scores.append(new_score)
        
    return new_gen, new_scores

if __name__ == "__main__":
    # Create constants
    Ny, Nx = 10, 10
    Nwall, Ntorch, Ndoor = 10, 5, 2
    Safety, Freedom = 1000, 1
    percent_eliminate, mutation_rate = .5, 5
    best_sc, best_w, t = 0, None, 0
    
    world_index = {}
    data = pd.DataFrame(columns=('ID', 'Score', 'Gen'))
    
    def save_data(gen, scores):
        """ Used to save the data at each iteration. """
        for world, sc in zip(gen, scores):
            name = uuid()
            world_index[name] = world
            data.loc[name] = {'Score': sc, 'Gen': t}
            draw(world, name='World{0}'.format(name), path='./GeneticLog/All/T{0}/'.format(t))
            
        # Get Best
        if max(scores) > best_sc:
            best_sc = max(scores)
            best_w  = gen[np.where(scores == best_sc)]
            draw(best_w, name='Score{0}'.format(best_sc), path='./GeneticLog/')
            
        data.to_csv('./GeneticLog/generations.csv')
            
    # Create a N random worlds
    generation, score_gen = [], []
    for n in N:
        # Generate new world with Nwall walls, Ntorch torches, and Ndoor doors
        new_world = random_world(Nx, Ny, original = None, items = [KEY['wall']], Ni = Nwall)
        new_world = random_world(Nx, Ny, original = new_world, items = [KEY['torch']], Ni = Ntorch)
        new_world = random_world(Nx, Ny, original = new_world, items = [KEY['door']], Ni = Ndoor)
        
        # Score the world
        new_score = simple_score(new_world, safety_weight = Safety, freedom_weight = Freedom)
        
        # Add to generation
        generation.append(new_world)
        score_gen.append(new_score)
    
    # Repeat Forever
    while True:
        # Save generation data
        save_data(generation, score_gen)
        
        # Increment time step
        t += 1
        
        # Eliminate percent_eliminate best samples based on their score (randomly, some that are weak should survive)
        # Breed them together and mutate the children
        generation, score_gen = resample(generation, score_gen, percent_eliminate = percent_eliminate, mutation_rate = mutation_rate)
        