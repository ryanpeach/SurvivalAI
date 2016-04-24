# ref: http://www.danielslater.net/2016/03/deep-q-learning-pong-with-tensorflow.html
# ref: http://colah.github.io/posts/2014-07-Conv-Nets-Modular/
# ref: Playing Atari with Deep Reinforcement Learning

import numpy as np
import tensorflow as tf
from Player2D import Player2D
from World2D import *
from collections import deque
import os, pdb, random
from copy import deepcopy

def hotone(index, L):
    out = np.zeros(L)
    out[index] = 1
    return out

def variable_summaries(var, name):
    """SOURCE: https://www.tensorflow.org/versions/r0.8/how_tos/summaries_and_tensorboard/index.html"""
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)

def in_layer(Nx = 84, Ny = 84, Ch = 4):
    # Input and output classes
    x = tf.placeholder("float", shape=[None, Nx, Ny, Ch])
    
    input_layer = tf.placeholder("float", [None, Nx, Ny, Ch])
    
    return input_layer

def visualNetwork2D(x, Nx = 84, Ny = 84, Ch = 4, Na = 12, 
                        c1s = 8, c1f = 16, c2s = 4, c2f = 32, m1s = 4, m2s = 2):
    """ Returns a shape of 1600, given [None, 80, 80, 1] """
    assert(Nx/m1s == int(Nx/m1s) and (Nx/m1s/m2s == int(Nx/m1s/m2s)))
    assert(Ny/m1s == int(Ny/m1s) and (Ny/m1s/m2s == int(Ny/m1s/m2s)))
    
    # Layer shape is [None, 80, 80, 1]
    with tf.name_scope("Convolution1"):
        with tf.name_scope("weights"):
            conv_w_1 = tf.Variable(tf.truncated_normal([c1s,c1s,Ch,c1f], stddev=0.01))
            variable_summaries(conv_w_1, "Convolution1" + '/weights')
        with tf.name_scope("biases"):
            conv_b_1 = tf.Variable(tf.truncated_normal([c1f], stddev=0.01))
            variable_summaries(conv_b_1, "Convolution1" + '/biases')
        with tf.name_scope('Wx_plus_b'):
            conv1_act = tf.nn.conv2d(x, conv_w_1, strides=[1, 1, 1, 1], padding='SAME') + conv_b_1
            tf.histogram_summary("Convolution1" + '/activations', conv1_act)
        conv1 = tf.nn.relu(conv1_act, 'relu')
        tf.histogram_summary("Convolution1" + '/activations_relu', conv1)
        
    with tf.name_scope("Max1"):
        max1 = tf.nn.max_pool(conv1, ksize=[1, m1s, m1s, 1], strides=[1, m1s, m1s, 1], padding='SAME')
        tf.histogram_summary("Max1", max1)
        
    # Layer shape is [None, 20, 20, 32]
    with tf.name_scope("Convolution2"):
        with tf.name_scope("weights"):
            conv_w_2 = tf.Variable(tf.truncated_normal([c2s,c2s,c1f,c2f], stddev=0.01))
            variable_summaries(conv_w_2, "Convolution2" + '/weights')
        with tf.name_scope("biases"):
            conv_b_2 = tf.Variable(tf.truncated_normal([c2f], stddev=0.01))
            variable_summaries(conv_b_2, "Convolution2" + '/biases')
        with tf.name_scope('Wx_plus_b'):
            conv2_act = tf.nn.conv2d(max1, conv_w_2, strides=[1, 1, 1, 1], padding='SAME') + conv_b_2
            tf.histogram_summary("Convolution2" + '/activations', conv2_act)
        conv2 = tf.nn.relu(conv2_act, 'relu')
        tf.histogram_summary("Convolution2" + '/activations_relu', conv2)

    with tf.name_scope("Max2"):
        max2 = tf.nn.max_pool(conv2, ksize=[1, m2s, m2s, 1], strides=[1, m2s, m2s, 1], padding='SAME')
        tf.histogram_summary("Max2", max2)
        
    return tf.reshape(max2, [-1, int((Nx/m1s/m2s)*(Ny/m1s/m2s)*c2f)]) # Layer shape [None, 5, 5, 64] 1600 Total
    
def ffNetwork(x, No = 512, Na = 12):
    """ Copied from http://www.danielslater.net/2016/03/deep-q-learning-pong-with-tensorflow.html, making modifications"""
    
    with tf.name_scope('FF1'):
        with tf.name_scope('weights'):
            ff_w_1 = tf.Variable(tf.truncated_normal([x.get_shape().as_list()[1], No], stddev=0.01))
            variable_summaries(ff_w_1, "FF1" + '/weights')
        with tf.name_scope('biases'):
            ff_b_1 = tf.Variable(tf.constant(0.01, shape=[No]))
            variable_summaries(ff_b_1, "FF1" + '/biases')
        with tf.name_scope('Wx_plus_b'):
            ff1_act = tf.matmul(x, ff_w_1) + ff_b_1
            tf.histogram_summary("FF1" + '/activations', ff1_act)
        ff1 = tf.nn.relu(ff1_act, 'relu')
        tf.histogram_summary("FF1" + '/activations_relu', ff1)
        
    with tf.name_scope('FF2'):
        with tf.name_scope('weights'):
            ff_w_2 = tf.Variable(tf.truncated_normal([No, Na], stddev=0.01))
            variable_summaries(ff_w_2, "FF2" + '/weights')
        with tf.name_scope('biases'):
            ff_b_2 = tf.Variable(tf.constant(0.01, shape=[Na]))
            variable_summaries(ff_b_2, "FF2" + '/biases')
        out = tf.matmul(ff1, ff_w_2) + ff_b_2
        tf.histogram_summary("FF2" + '/activations_relu', out)
    
    return out

def qGenerator():
    """ Used as an iterative action selector in reinforcement learning.
        Takes a shape [-1, 1600] and returns a shape [-1, 13] """
    raise NotImplementedError

def denseGenerator():
    """ Used as a creative generator.
        Takes a shape [None, Nx, Ny, Nz] and returns a shape [None, Nx, Ny, Nz]. """
    raise NotImplementedError

class RollingMemory(object):
    """ A class used for memory which pops its oldest element after
        reaching a certain size """
    def __init__(self, size):
        self.Mem = deque()
        self.MAX = size
        
    def __getitem__(self, i):
        return self.Mem[i]
    
    def __setitem__(self, i, v):
        self.Mem[i] = v
        
    def add(self, i):
        self.Mem.append(i)
        if len(self.Mem)>self.MAX:
            return self.Mem.popleft()
        else:
            return None
            
    def copy(self):
        return deepcopy(self.Mem)
        
    def __iter__(self):
        return iter(self.Mem)
    
    def __len__(self):
        return len(self.Mem)
        
class QPlayer2D(Player2D):
    """ My own version of the class """
    FAILURE_COST = -1
    SAFETY_WEIGHT  = 1000
    FREEDOM_WEIGHT = 1

    def __init__(self, world, start_loc, inv, Nt = 1, Nz = 1, 
                       learn_rate = .9, future_weight = .9, memsize = 1e9, save_freq = 1e3,
                       path = './nn/', realtime = True):
        super(QPlayer2D, self).__init__(world, start_loc, inv, realtime)
        
        # Set variables
        self._path = path
        self.Nt, self.Nz = Nt, Nz
        self.Ch = Nt*Nz
        self.T = 0
        self.best_sc = -100000
        self.future_weight = .9
        self.save_freq = save_freq
        self.MEM_SIZE = memsize
        
        # Create your memory
        self.memory = RollingMemory(memsize)
        self.working_memory = np.ones([self.Nx,self.Ny,Nt])
        self.update_working_memory()
        self.last_reward = 0
        
        # Create session and network
        self._session = tf.Session()
        self._x, self._y = self._create_network()
        
        # Create training network
        self._action = tf.placeholder("float", [None, self.Na])
        self._target = tf.placeholder("float", [None])

        with tf.name_scope('Test'):
            with tf.name_scope('readout'):
                readout_action = tf.reduce_sum(tf.mul(self._y, self._action), reduction_indices=1)
            with tf.name_scope('cost'):
                cost = tf.reduce_mean(tf.square(self._target - readout_action))
            with tf.name_scope('train'):
                self._train_operation = tf.train.AdamOptimizer(learn_rate).minimize(cost)
        
        # Run Session
        train_writer = tf.train.SummaryWriter(self._path + '/tfboatd', self._session.graph)
        self._session.run(tf.initialize_all_variables())
        
        # Create a saver and load data
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
        self.T += 1                             # Increment time
        succ = self.action_list[action_index]() # Perform action
        if succ:                                # Check for successful action
            reward = self._score()              # If successful, get score
            dr = reward - self.last_reward      # Get the derivative
            self.last_reward = reward           # Update last reward
        else:                                   # If not successful
            reward = self.last_reward           # No need to recalculate
            dr = self.FAILURE_COST              # difference is 0
            
        # Set best score
        if reward > self.best_sc:
            print("Best Score: {0}".format(reward))
            print("Time: {0}".format(self.T))
            self.best_sc = reward
            self._display('Score{0}'.format(abs(reward)))
                
        # Return score difference
        return dr
        
    def action_randomize(self, action_index):
        # Set our confidence linearly from .1 to .9
        a = .1
        m, b = -(1.-a)/self.MEM_SIZE, 1.0
        if self.training:
            if self.T < self.MEM_SIZE:
                conf = m*self.T + b
            else:
                conf = a
        else:
            conf = a
        
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
        
        # Create new state
        newstate = [prior, hotone(action_index, self.Na), reward, self.working_memory.copy()]
        
        # Return newstate
        return newstate
    
    def get_memory(self):
        return self.memory.copy()
    
    def run(self, N = 1000, training = True, Ns = 5):
        self.training = training
        for n in range(N):
            new_state = next(self)
            # Add terminal information
            if n == N-1:
                new_state.append(True)
            else:
                new_state.append(False)
            
            # Add new_state to memory
            self.memory.add(tuple(new_state))
            
            # Perform a training step if training
            if training:
                self.train(Ns)
                
        return True
        
    def train(self, Ns):
        # Get a random sample of the memory
        if Ns > len(self.memory):
            Ns = len(self.memory)
        mem_sample_index = random.sample(tuple(range(len(self.memory))), Ns)    # Get random sample of indexes
        states, actions, rewards, results, terminal = [], [], [], [], []
        for i in mem_sample_index:                                              # Retrieve those indexes from memory 
            states.append(self.memory[i][0])
            actions.append(self.memory[i][1])
            rewards.append(self.memory[i][2])
            results.append(self.memory[i][3])
            terminal.append(self.memory[i][4])
            
        # Get predicted rewards for each result as is
        reward_predictions = self._session.run(self._y, feed_dict={self._x: results})
        expected_rewards = []
        for i in range(Ns):
            if terminal[i]:
                expected_rewards.append(rewards[i])
            else:
                expected_rewards.append(rewards[i] + self.future_weight * 
                    np.max(reward_predictions[i])) # This is the Bellman Equation
            
        # learn that these actions in these states lead to this reward
        self._session.run(self._train_operation, feed_dict={
            self._x: states,
            self._action: actions,
            self._target: expected_rewards})

        # save checkpoints for later
        if self.T % self.save_freq == 0:
            self._saver.save(self._session, self._path + '/network', global_step=self.T)
           
     
if __name__=="__main__":
    world = random_world(Nx = 12, Ny = 12, items = [KEY['wall']], Ni = 10)
    world = random_world(Nx = 12, Ny = 12, original = world, items = [KEY['torch']], Ni = 3)
    start_loc = (np.random.randint(0,12), np.random.randint(0,12))
    inv   = {KEY['wall']: 50, KEY['torch']: 10, KEY['door']: 2}
    player = QPlayer2D(world, start_loc, inv, Nt = 4, Nz = 1, 
                       learn_rate = .9, future_weight = .9, memsize = 1e9,
                       path = './nn/', realtime = True)
                       
    def reset_world():
        # Reset World
        world = random_world(Nx = 12, Ny = 12, items = [KEY['wall']], Ni = 10)
        world = random_world(Nx = 12, Ny = 12, original = world, items = [KEY['torch']], Ni = 3)
        player.W = world
        player.INV = {KEY['wall']: 50, KEY['torch']: 10, KEY['door']: 2}
        
    def run_once():
        # Run and train on the data
        player.run(int(1e3), training = True, Ns = 1e3)
        
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
        run_once()
        save_scores()
    
    # Then, we want to worry about the amount of freedom our house gives
    player.FREEDOM_WEIGHT = 10        
    while True:
        reset_world()
        for j in range(10):
            run_once()
        save_scores()
