from World2D import *

from uuid import uuid4 as uuid
import os

import pandas as pd
import numpy as np
from numpy.random import choice, randint


GENES = [KEY['wall'], KEY['torch'], KEY['door'], KEY['space']]
GENES_FREQ = [1,1,1,3]
GENES_FREQ = list(np.array(GENES_FREQ) / np.sum(GENES_FREQ))    # Normalize frequency
COST = {KEY['wall']: 4, KEY['torch']: 5, KEY['door']: 16, KEY['space']: 4}

def combine(A, B):
    # Check type and get variables
    assert(A.shape == B.shape)
    Ny, Nx = A.shape
    Na, Nb = len(np.where(A != KEY['space'])[0]), len(np.where(B != KEY['space'])[0])
    
    # Copy A and B to prevent overriding data
    A, B = A.copy(), B.copy()

    # Generate coordinate array to sample
    X, Y = np.meshgrid(np.arange(Ny),np.arange(Nx))
    COORD = np.dstack((X,Y))
    COORD = np.reshape(COORD, [Ny*Nx,2])
    
    Ca = np.vectorize(lambda a: COST[a])(A)  # Cost matrix for A
    Cb = np.vectorize(lambda b: COST[b])(B)  # Cost matrix for B
    return np.vectorize(lambda a,b,ca,cb: choice([a,b], size=1, p=[1-ca/(ca+cb),1-cb/(ca+cb)]))(A,B,Ca,Cb)
    
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
    assert(len(gen)==len(scores))           # They must be the same length
    N = len(gen)                               # N is equal to that length
    Ns = int(len(gen)*(1.-percent_eliminate))  # Get the sample size
    P = (np.array(scores, dtype='float')+abs(min(scores))+.001) / np.sum(np.array(scores)+abs(min(scores))+.001) # normalize scores to get probabilities

    # Kill off percent_eliminate of the worst worlds
    survive_index = choice(np.arange(len(gen)), size = Ns, replace = False, p = P)
    gen, scores = [gen[i] for i in survive_index], [scores[i] for i in survive_index]
    P = (np.array(scores, dtype='float')+abs(min(scores))+.001) / np.sum(np.array(scores)+abs(min(scores))+.001) # normalize scores to get probabilities
    
    # Sample the generation based on score, higher the more probable to breed
    A = choice(np.arange(Ns), size = N, replace = True, p = P)
    B = choice(np.arange(Ns), size = N, replace = True, p = P)
    A, B = [gen[i] for i in A], [gen[i] for i in B]
    
    # Generate the next generation
    new_gen, new_scores = [], []
    for a, b in zip(A,B):
        # Generate new world via gene combination and mutation
        new_world = combine(a,b)
        new_world = mutate(new_world, rate = mutation_rate)
        
        # Score the new world
        new_score = score(new_world, safety_weight = Safety, freedom_weight = Freedom)
        
        # Append it to the new generation
        new_gen.append(new_world)
        new_scores.append(new_score)
        
    return new_gen, new_scores

def score(W, safety_weight = 1000, freedom_weight = 1):
    L = generate_light(W, d_l = 2)
    P = generate_particles(W, L, p = KEY['parti'])
    S = run_simulation(P, p = KEY['parti'], impassable = enemy_not_passable, fill = KEY['parti'])
    C = run_simulation(W, p = -1, impassable = not_passable, fill = -1)
    world_score = scoreWorld(S,C,safety_weight = safety_weight, freedom_weight = freedom_weight)
    
    w = COST[KEY['wall']]*np.sum(W == KEY['wall'])
    t = COST[KEY['torch']]*np.sum(W == KEY['torch'])
    d = COST[KEY['door']]*np.sum(W == KEY['door'])
    
    return world_score - w - t - d
    
if __name__ == "__main__":
    # Create constants
    N = 10
    Ny, Nx = 12, 12
    Nwall, Ntorch, Ndoor = 100, 20, 2
    Safety, Freedom = 1000., 10.
    percent_eliminate, mutation_rate = .5, 5
    best_sc, best_w, t = 0, None, 0
    
    world_index, score_index = {}, {}
    data = pd.DataFrame(columns=('Score', 'Gen'))
    
    def create_path(path):
        if not os.path.exists(path):
            os.makedirs(path)
            
    def save_data(gen, scores):
        """ Used to save the data at each iteration. """
        global best_sc
        for world, sc in zip(gen, scores):
            # Index world and save to spreadsheet
            name = uuid()
            world_index[name] = world
            score_index[name] = sc
            data.loc[name] = {'Score': sc, 'Gen': t}
            
            # Draw world
            draw_path = './GeneticLog/All/T{0}/'.format(t)
            create_path(draw_path)
            draw(world, name='World{0}'.format(name), path=draw_path)
            
        # Get Best
        if max(scores) > best_sc:
            best_sc = max(scores)
            best_w  = gen[np.where(scores == best_sc)[0][0]]
            
            # Draw Best at root
            create_path('./GeneticLog/')
            draw(best_w, name='Score{0}'.format(best_sc), path='./GeneticLog/')
            print("New Best: {1} T: {0}".format(t,best_sc))
        
        # Overwrite Spreadsheet
        create_path('./GeneticLog/')
        data.to_csv('./GeneticLog/generations.csv')
        
        # Print every 100 iterations
        if t % 10 == 0:
            print("T: {0}, Best Score: {1}".format(t,best_sc))
            
    # Create a N random worlds
    generation, score_gen = [], []
    for n in np.arange(N):
        # Generate new world with Nwall walls, Ntorch torches, and Ndoor doors
        new_world = random_world(Nx, Ny, original = None, items = [KEY['wall']], Ni = Nwall)
        new_world = random_world(Nx, Ny, original = new_world, items = [KEY['torch']], Ni = Ntorch)
        new_world = random_world(Nx, Ny, original = new_world, items = [KEY['door']], Ni = Ndoor)
        
        # Score the world
        new_score = score(new_world, safety_weight = Safety, freedom_weight = Freedom)
        
        # Add to generation
        generation.append(new_world)
        score_gen.append(new_score)
    
    # Repeat Forever
    for i in range(10):
        # Save generation data
        save_data(generation, score_gen)
        
        # Increment time step
        t += 1
        
        # Eliminate percent_eliminate best samples based on their score (randomly, some that are weak should survive)
        # Breed them together and mutate the children
        generation, score_gen = resample(generation, score_gen, percent_eliminate = percent_eliminate, mutation_rate = mutation_rate)
    
    from AnalogyBuilder import ScoreRegression
    worlds, scores = [], []
    for name in world_index:
        worlds.append(world_index[name])
        scores.append(score_index[name])
        
    world, sc = ScoreRegression(worlds, scores)
    draw(world[0,:,:], name='NNWorld', path='./GeneticLog/')
    test_sc = score(world[0,:,:])
    print("Test Score: {0}".format(test_sc))