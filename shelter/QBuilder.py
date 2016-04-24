# ref: http://www.danielslater.net/2016/03/deep-q-learning-pong-with-tensorflow.html
# ref: http://colah.github.io/posts/2014-07-Conv-Nets-Modular/
# ref: Playing Atari with Deep Reinforcement Learning

import numpy as np
import tensorflow as tf
from Player2D import Player2D
from World2D import *
from collections import deque
import os, pdb, random

def hotone(index, L):
    out = np.zeros(L)
    out[index] = 1
    return out

def in_layer(Nx = 80, Ny = 80, Ch = 1):
    # Input and output classes
    x = tf.placeholder("float", shape=[None,Nx,Ny,Ch])
    
    input_layer = tf.placeholder("float", [None, Nx, Ny, Ch])
    
    return input_layer

def visualNetwork2D(x, Nx = 12, Ny = 12, Ch = 1, Na = 12, 
                        c1s = 3, c1f = 32, c2s = 2, c2f = 64, m1s = 2, m2s = 2):
    """ Returns a shape of 1600, given [None, 80, 80, 1] """
    assert(Nx/m1s == int(Nx/m1s) and (Nx/m1s/m2s == int(Nx/m1s/m2s)))
    assert(Ny/m1s == int(Ny/m1s) and (Ny/m1s/m2s == int(Ny/m1s/m2s)))
    
    # Layer shape is [None, 80, 80, 1]
    conv_w_1 = tf.Variable(tf.truncated_normal([c1s,c1s,Ch,c1f], stddev=0.01))
    conv_b_1 = tf.Variable(tf.truncated_normal([c1f], stddev=0.01))
    conv1    = tf.nn.relu(tf.nn.conv2d(x, conv_w_1, strides=[1, 1, 1, 1], padding='SAME') + conv_b_1)
    
    max1 = tf.nn.max_pool(conv1, ksize=[1, m1s, m1s, 1], strides=[1, m1s, m1s, 1], padding='SAME')

    # Layer shape is [None, 20, 20, 32]
    conv_w_2 = tf.Variable(tf.truncated_normal([c2s,c2s,c1f,c2f], stddev=0.01))
    conv_b_2 = tf.Variable(tf.truncated_normal([c2f], stddev=0.01))
    conv2    = tf.nn.relu(tf.nn.conv2d(max1, conv_w_2, strides=[1, 1, 1, 1], padding='SAME') + conv_b_2)

    max2 = tf.nn.max_pool(conv2, ksize=[1, m2s, m2s, 1], strides=[1, m2s, m2s, 1], padding='SAME')
    
    return tf.reshape(max2, [-1, int((Nx/m1s/m2s)*(Ny/m1s/m2s)*c2f)]) # Layer shape [None, 5, 5, 64] 1600 Total
    
def ffNetwork(x, No = 512, Na = 12):
    """ Copied from http://www.danielslater.net/2016/03/deep-q-learning-pong-with-tensorflow.html, making modifications"""
    ff_w_1 = tf.Variable(tf.truncated_normal([x.get_shape().as_list()[1], No], stddev=0.01))
    ff_b_1 = tf.Variable(tf.constant(0.01, shape=[No]))
    ff_w_2 = tf.Variable(tf.truncated_normal([No, Na], stddev=0.01))
    ff_b_2 = tf.Variable(tf.constant(0.01, shape=[Na]))
    
    ff1 = tf.nn.relu(tf.matmul(x, ff_w_1) + ff_b_1)
    out = tf.matmul(ff1, ff_w_2) + ff_b_2
    
    return out

def qGenerator():
    """ Used as an iterative action selector in reinforcement learning.
        Takes a shape [-1, 1600] and returns a shape [-1, 13] """
    raise NotImplementedError

def denseGenerator():
    """ Used as a creative generator.
        Takes a shape [None, Nx, Ny, Nz] and returns a shape [None, Nx, Ny, Nz]. """
    raise NotImplementedError


class QPlayer2D(Player2D):
    """ My own version of the class """
    PERFORMACE_COST = -1
    FAILURE_COST    = -2
    EXPLORE_STEPS   = 1e10
    SAFETY_WEIGHT = 1000
    FREEDOM_WEIGHT = 1

    def __init__(self, world, start_loc, inv, Nt = 1, Nz = 1, 
                       learn_rate = .9, path = './nn/', realtime = True):
        super(QPlayer2D, self).__init__(world, start_loc, inv, realtime)
        
        self._path = path
        self.Nt, self.Nz = Nt, Nz
        self.Ch = Nt*Nz
        self.T = 0
        self.best_sc = -100000
        self.working_memory = np.ones([self.Nx,self.Ny,Nt])
        self.update_working_memory()
        self.last_reward = 0
        
        self._session = tf.Session()
        self._x, self._y = self._create_network()
        
        self.reward_list = [self._score]
                            
        self._action = tf.placeholder("float", [None, self.Na])
        self._target = tf.placeholder("float", [None])

        readout_action = tf.reduce_sum(tf.mul(self._y, self._action), reduction_indices=1)

        cost = tf.reduce_mean(tf.square(self._target - readout_action))
        self._train_operation = tf.train.AdamOptimizer(learn_rate).minimize(-cost)
        
        self._session.run(tf.initialize_all_variables())
        
        if not os.path.exists(self._path):
            os.mkdir(self._path)
        self._saver = tf.train.Saver()
        load_data = tf.train.get_checkpoint_state(self._path)

        if load_data and load_data.model_checkpoint_path:
            self._saver.restore(self._session, load_data.model_checkpoint_path)
            print("Loaded checkpoints %s" % load_data.model_checkpoint_path)
    
    def _score(self):
        L = generate_light(self.W, d_l = 2)
        P = generate_particles(self.W, L, p = KEY['parti'])
        S = run_simulation(P)   # Run simulation on the particles
        
        D = self.W.copy()
        D[self.Loc] = -1
        C = run_simulation(D, p = -1, impassable = not_passable, fill = KEY['space'])
        return scoreWorld(self.W, S, C, self.SAFETY_WEIGHT, self.FREEDOM_WEIGHT)
        
    def update_working_memory(self):
        self.working_memory = np.roll(self.working_memory, 1, axis=2)
        self.working_memory[:,:,0] = self.W
        
    def _create_network(self):
        input_layer = in_layer(Nx = self.Nx, Ny = self.Ny, Ch = self.Ch)
        conv = visualNetwork2D(input_layer, Nx = self.Nx, Ny = self.Ny, Ch = self.Ch, Na = self.Na, 
                        c1s = 3, c1f = 32, c2s = 2, c2f = 64, m1s = 2, m2s = 2)
        output_layer = tf.nn.softmax(ffNetwork(conv, No = 512, Na = self.Na))
        
        return input_layer, output_layer
        
    def action_reward(self, action_index):
        """ Performs the action at the given index and returns a reward. """
        self.T += 1
        succ = self.action_list[action_index]()
        if succ:
            reward = self.reward_list[0]()
            dr = reward - self.last_reward
            self.last_reward = reward
            if reward > self.best_sc:
                print("Best Score: {0}".format(reward))
                print("Time: {0}".format(self.T))
                self.best_sc = reward
                self._display('Score{0}'.format(abs(reward)))
        else:
            dr = self.FAILURE_COST
        return dr + self.PERFORMACE_COST
        
    def action_randomize(self, action_index):
        # Set our confidence linearly from .1 to .9
        if self.training:
            m, b = .8/self.EXPLORE_STEPS, .1
            conf = m*self.T + b
        else:
            conf = 0.01
        
        # Select random action if dice comes up greater than ever increasing confidence
        if np.random.rand() > conf:
            return np.random.randint(0, self.Na)
        else:
            return action_index
        
    def __next__(self):
        """ This runs the agent forward one timestep. """

        # Run and get a reward
        values = self._session.run(self._y, feed_dict={self._x: [self.working_memory]})[0]
        action_index = self.action_randomize(np.argmax(values))
        
        # Save Prior
        prior = self.working_memory.copy()
        
        # Perform Action
        reward = self.action_reward(action_index)
        
        # Update Working Memory
        self.update_working_memory()
        
        # Return new state
        return (prior, hotone(action_index, self.Na), reward, self.working_memory.copy())
    
    def run(self, N = 1000, training = True):
        self.training = training
        memory = deque()
        for i in range(N):
            new_state = next(self)
            memory.append(new_state)
        
        return memory
        
    def train(self, memory, Ns = 5, future_weight = .9, save = True):
        # Get a random sample of the memory
        mem_sample_index = random.sample(tuple(range(len(memory))), Ns)
        states, actions, rewards, results = [], [], [], []
        for i in mem_sample_index:
            states.append(memory[i][0])
            actions.append(memory[i][1])
            rewards.append(memory[i][2])
            results.append(memory[i][3])
            
        # Get predicted rewards for each result as is
        reward_predictions = self._session.run(self._y, feed_dict={self._x: results})
        expected_rewards = []
        for i in range(Ns):
            expected_rewards.append(rewards[i] + future_weight * np.max(reward_predictions[i])) # This is the Bellman Equation
            
        # learn that these actions in these states lead to this reward
        self._session.run(self._train_operation, feed_dict={
            self._x: states,
            self._action: actions,
            self._target: expected_rewards})

        # save checkpoints for later
        if save:
            self._saver.save(self._session, self._path + '/network', global_step=self.T)
           
     

if __name__=="__main__":
    world = random_world(Nx = 12, Ny = 12, items = [KEY['wall']], Ni = 10)
    world = random_world(Nx = 12, Ny = 12, original = world, items = [KEY['torch']], Ni = 3)
    start_loc = (np.random.randint(0,12), np.random.randint(0,12))
    inv   = {KEY['wall']: 50, KEY['torch']: 10, KEY['door']: 2}
    player = QPlayer2D(world, start_loc, inv, Nt = 5, Nz = 1, 
                       learn_rate = .9, path = './nn/', realtime = True)
    def reset_world():
        # Reset World
        world = random_world(Nx = 12, Ny = 12, items = [KEY['wall']], Ni = 10)
        world = random_world(Nx = 12, Ny = 12, original = world, items = [KEY['torch']], Ni = 3)
        player.W = world
        player.INV = {KEY['wall']: 50, KEY['torch']: 10, KEY['door']: 2}
        
    def run_once():
        # Run and train on the data
        memory = player.run(int(1e3), training = True)
        player.train(memory, Ns = int(5e2), future_weight = .9, save = True)
        
    def save_scores():
        # Handle Score
        scores.append(player.best_sc)
        player.best_sc = -10000
        
        # Plot scores
        plt.figure()
        plt.plot(scores)
        plt.savefig('./log/scores.png')
        plt.close()
        print("Regenerating World")
                
    scores = []
    
    # At first we just want to score houses
    player.SAFETY_WEIGHT = 1000
    player.FREEDOM_WEIGHT = 0
    for i in range(1000):
        reset_world()
        for j in range(10):
            run_once()
        save_scores()
    
    # Then, we want to worry about the amount of freedom our house gives
    player.FREEDOM_WEIGHT = 1        
    while True:
        reset_world()
        for j in range(10):
            run_once()
        save_scores()
