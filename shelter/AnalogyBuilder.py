import tensorflow as tf
from World2D import *

COST = {KEY['wall']: 4, KEY['torch']: 5, KEY['door']: 16, KEY['space']: 4}

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

def world_to_tensor(world_class):
    with tf.name_scope('World_to_Tensor'):
        Ny, Nx = world_class.shape[0:2]
        world_class = np.resize(world_class, [1,Ny,Nx,1])
    
        # Get object locations
        torches = np.where(world_class == KEY['torch'])
        space   = np.where(world_class == KEY['space'])
        door    = np.where(world_class == KEY['door'])
        wall    = np.where(world_class == KEY['wall'])
        
        # Create a key representation of the world
        key_world = np.zeros([Ny,Nx])
        key_world[torches[1:2]] = KEY['torch']
        key_world[space[1:2]]   = KEY['space']
        key_world[door[1:2]]    = KEY['door']
        key_world[wall[1:2]]    = KEY['wall']
        
        # Set enemy passable channels 
        enemy_passable    = np.full([1, Ny, Nx, 1], fill_value = False)
        enemy_passable[torches] = True
        enemy_passable[space]   = True
        
        # Set player passable channels
        player_passable   = np.full([1, Ny, Nx, 1], fill_value = False)
        player_passable[space]   = True
        player_passable[torches] = True
        player_passable[door]    = True
        
        # Set light source channels
        light_source      = np.full([1, Ny, Nx, 1], fill_value = False)
        light_source[torches] = True
        
        # Set cost channels
        cost          = np.zeros([1, Ny, Nx, 1])
        cost[torches] = COST[KEY['torch']]
        cost[door]    = COST[KEY['door']]
        cost[space]   = COST[KEY['space']]
        cost[wall]    = COST[KEY['wall']]
        
        # Get accessiblity channels
        L = generate_light(key_world, d_l = 2)
        P = generate_particles(key_world, L, p = KEY['parti'])
        S = run_simulation(P, p = KEY['parti'], impassable = enemy_not_passable, fill = KEY['parti'])
        C = run_simulation(key_world, p = -1, impassable = not_passable, fill = -1)
        
        # Set enemy accessiblity
        enemy_accessible  = np.full([1, Ny, Nx, 1], fill_value = False)
        enemy = np.where(S == KEY['parti'])
        enemy_accessible[0, enemy[0], enemy[1], 0] = True
        
        # Set player accessiblity
        player_accessible = np.full([1, Ny, Nx, 1], fill_value = False)
        player = np.where(C == -1)
        player_accessible[0, player[0], player[1], 0] = True
        
        # Set Light
        light = np.full([1, Ny, Nx, 1], fill_value = False)
        lloc = np.where(L == 1)
        light[0, lloc[0], lloc[1], 0] = True
        
        # Generate Tensor
        world_data = np.concatenate([enemy_passable, player_passable, light_source, cost, enemy_accessible, player_accessible, light], axis = 3)
        
        return world_data.astype('float')

def score_tensor(world, safety_weight, freedom_weight):
    with tf.name_scope('Score_tensor'):
        S = tf.cast(world[:,:,4],dtype='bool')
        C = tf.cast(world[:,:,5],dtype='bool')
        
        # Get the count of safe spaces
        safe = tf.logical_and(C, tf.logical_not(S))
        safe = tf.reduce_sum(tf.cast(safe, dtype='float'))
        
        # Get the count of free spaces
        free = tf.reduce_sum(tf.cast(C, dtype='float'))
        return safe*safety_weight + free*freedom_weight


def feature_extraction(x, Nx, Ny, Ch, conv_w_1, conv_b_1, conv_w_2, conv_b_2, ff_w_1, ff_b_1, m1s = 4, m2s = 2, No = 512):
    """ Creates a convolutional neural network to analyze some world tensor and return features from it. """
    # Check that all the sizes are consistent
    assert(Nx/m1s == int(Nx/m1s) and (Nx/m1s/m2s == int(Nx/m1s/m2s)))
    assert(Ny/m1s == int(Ny/m1s) and (Ny/m1s/m2s == int(Ny/m1s/m2s)))
    
    # First Convolutional Layer
    with tf.name_scope("Convolution1"):
        conv1_act = tf.nn.conv2d(x, conv_w_1, strides=[1, 1, 1, 1], padding='SAME') + conv_b_1
        conv1 = tf.nn.relu(conv1_act, 'relu')
        
    # First Max Pooling Layer
    with tf.name_scope("Max1"):
        max1 = tf.nn.max_pool(conv1, ksize=[1, m1s, m1s, 1], strides=[1, m1s, m1s, 1], padding='SAME')
        
    # Second Convolutional Layer
    with tf.name_scope("Convolution2"):
        conv2_act = tf.nn.conv2d(max1, conv_w_2, strides=[1, 1, 1, 1], padding='SAME') + conv_b_2
        conv2 = tf.nn.relu(conv2_act, 'relu')

    # Second Max Pooling Layer
    with tf.name_scope("Max2"):
        max2 = tf.nn.max_pool(conv2, ksize=[1, m2s, m2s, 1], strides=[1, m2s, m2s, 1], padding='SAME')

    # Reshaping max2 for FF1
    max2_rshp = tf.reshape(max2, [-1, 288]) # Layer shape [None, 5, 5, 64] 1600 Total
    
    # First Feed Forward Layer
    with tf.name_scope('FF1'):
        ff1_act = tf.matmul(max2_rshp, ff_w_1) + ff_b_1
        ff1 = tf.nn.relu(ff1_act, 'relu')
    
    return ff1

def key_to_tensor(W, Nobjects):
    vec = lambda y, x: np.array([1. if i == W[y,x] else 0. for i in np.arange(Nobjects)])
    new_tensor = np.zeros((W.shape[0],W.shape[1],Nobjects))
    for y in np.arange(W.shape[0]):
        for x in np.arange(W.shape[1]):
            new_tensor[y,x,:] = vec(y,x)
    return new_tensor


# Channels: Enemy_Accessible, Character_Accessible, Light, Cost
# Attributes: Enemy_passable, Character_passable, Light-Source
def ScoreRegression(worlds, scores, OPS = ['train','score']):
    Nx, Ny = worlds[0].shape
    Nobjects, Ch, No = 4, 4, 512
    worlds = [key_to_tensor(x, Nobjects) for x in worlds]
    scores = [np.reshape(x, [1]) for x in scores]
    
    with tf.Session() as sess:
        with tf.name_scope("Declarations"):
            c1s, c1f, c2s, c2f = 3, 16, 2, 32
            W_in  = tf.placeholder(shape = [None, Nx, Ny, Nobjects], dtype = 'float', name = "Worlds")
            conv_w_1 = tf.Variable(tf.truncated_normal([c1s,c1s,Ch,c1f], stddev=1.01))
            conv_b_1 = tf.Variable(tf.truncated_normal([c1f], stddev=1.01))
            conv_w_2 = tf.Variable(tf.truncated_normal([c2s,c2s,c1f,c2f], stddev=1.01))
            conv_b_2 = tf.Variable(tf.truncated_normal([c2f], stddev=1.01))
            ff_w_1 = tf.Variable(tf.truncated_normal([288, No], stddev=1.01))
            ff_b_1 = tf.Variable(tf.constant(0.01, shape=[No]))
            ff_w_2 = tf.Variable(tf.truncated_normal([512, 1], stddev = 0.01))
            ff_b_2 = tf.Variable(tf.truncated_normal([1]))
            keep_prob = tf.placeholder(tf.float32)
            
        with tf.name_scope("World_Variable"):
            W_var = tf.Variable(tf.truncated_normal([1,Nx,Ny,Nobjects], stddev=5.01), name = "World_Var")

        with tf.name_scope('Regression_Network'):
            with tf.name_scope('Features'):
                ff1 = feature_extraction(W_in, Nx, Ny, Ch, conv_w_1, conv_b_1, conv_w_2, conv_b_2, ff_w_1, ff_b_1, m1s = 2, m2s = 2, No = No)
            with tf.name_scope('Regression'):
                with tf.name_scope('FF2'):
                    with tf.name_scope('dropout'):
                        ff1_drop = tf.nn.dropout(ff1, keep_prob)
                    ff2_act = tf.matmul(ff1_drop, ff_w_2) + ff_b_2
                    ff2 = tf.nn.relu(ff2_act, 'relu')
        
        with tf.name_scope('Creation_Network'):
            with tf.name_scope('Features'):
                ff1_cre = feature_extraction(W_var, Nx, Ny, Ch, conv_w_1, conv_b_1, conv_w_2, conv_b_2, ff_w_1, ff_b_1, m1s = 2, m2s = 2, No = No)
            with tf.name_scope('Regression'):
                with tf.name_scope('FF2'):
                    with tf.name_scope('dropout'):
                        ff1_drop_cre = tf.nn.dropout(ff1_cre, keep_prob)
                    ff2_act_cre = tf.matmul(ff1_drop_cre, ff_w_2) + ff_b_2
                    ff2_cre = tf.nn.relu(ff2_act_cre, 'relu')
        
        trainer = tf.train.AdamOptimizer(.01)
        with tf.name_scope('Regression_Accuracy'):
            S = tf.placeholder(shape = [None, 1], dtype = 'float', name = "Scores")
            dist = tf.reduce_mean(tf.abs(ff2 - S), name = "Dist")
            tf.scalar_summary('regressor/score', dist)
            reg_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Declarations")
            train_regression = trainer.minimize(dist, var_list=reg_vars)

        with tf.name_scope('Creation_Score'):
            dist_w = tf.reduce_mean(tf.abs(ff2_cre), name = "Value")
            tf.scalar_summary('creator/score', dist_w)
            cr_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "World_Variable")
            gen_cr = trainer.minimize(-dist_w, var_list=cr_vars)
            
        # Create summary writer
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter('./log/',
                                        graph_def=sess.graph_def)
        
        # Initialize Variables
        sess.run(tf.initialize_all_variables())
        
        # Train the networ
        t = 0
        cst = 1000
        while cst > 400.0:
            gen, cst, summ = sess.run([train_regression, dist, summary_op], feed_dict = {W_in: worlds, S: scores, keep_prob: .99})
            summary_writer.add_summary(summ, t)
            print("Cost: {0}".format(cst))
            t += 1
        
        # Create a new world
        while cst < 50000:
            gen, cst, wrld, summ = sess.run([gen_cr, dist_w, W_var, summary_op], feed_dict = {W_in: worlds, S: scores, keep_prob: 1.})
            summary_writer.add_summary(summ, t)
            print("Cost: {0}".format(cst))
            #print("World: {0}".format(wrld))
            t += 1
        
        wrld = np.argmax(wrld, axis = 3)
        print("Final")
        print("Cost: {0}".format(cst))
        print(wrld)
        return wrld, cst

    
# World: Wall, Door, Torch
def FindAnalogy(inspiration, style):
    Nx, Ny = inspiration.shape
    Nobjects, Ch = 3, 1
    max_score = Nx*Ny*2
    
    inspiration = np.reshape(inspiration, [1, Nx, Ny, 1])
    style = np.reshape(style, [1, Nx, Ny, 1])
    
    with tf.Session() as sess:
        # Get our placeholders and world variable
        with tf.name_scope("Declarations"):
            world = tf.argmax(tf.Variable(tf.truncated_normal([1,Nx,Ny,Nobjects], stddev=0.01)), 3, name = "world_class")
            world = tf.cast(world, dtype='float')
            world = tf.reshape(world, [1, Nx, Ny, 1])
            inspiration_in = tf.placeholder(shape = [1, Nx, Ny, 1], dtype = 'float', name = "Analogy1")
            style_in = tf.placeholder(shape = [1, Nx, Ny, 1], dtype = 'float', name = "Analogy2")
            print(world, inspiration_in, style_in)

        # Setup Feature Extractors
        with tf.name_scope("Feature_Extraction"):
            world_in = tf.concat(0, [world, inspiration_in, style_in], name = "world_in")
            features = feature_extraction(world_in, Nx = Nx, Ny = Ny, Ch = Ch, c1s = 3, c1f = 16, c2s = 2, c2f = 32, m1s = 2, m2s = 2, No = 512)
            world_features, inspiration_features, style_features = features[0,:], features[1,:], features[2,:]
            
        # Score world
        #w_score = score_tensor(world_in[0,:,:,:], tf.constant(10.), tf.constant(1.))
        #i_score = score_tensor(world_in[1,:,:,:], tf.constant(10.), tf.constant(1.))
        #s_score = score_tensor(world_in[2,:,:,:], tf.constant(10.), tf.constant(1.))
        
        # Generate Cost functions
        with tf.name_scope('Train'):
            merge_features = inspiration_features + style_features
            dist_w = tf.reduce_mean(tf.square(world_features - merge_features))
            generate = tf.train.AdamOptimizer(.01).minimize(tf.Print(dist_w, [dist_w, world]))

        # Create summary writer
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter('./log/',
                                        graph_def=sess.graph_def)
        
        # Initialize Variables
        sess.run(tf.initialize_all_variables())
        
        # Training loop
        t = 0
        cst = 1000
        while cst > 0.0:
            _, summ = sess.run([generate, summary_op], feed_dict = {inspiration_in: inspiration, style_in: style})
            summary_writer.add_summary(summ, t)
            #print("Cost: {0}".format(cst))
            #print("World: {0}".format(wld[0,:,:,0]))
            t += 1
            
        return sess.run(world)

if __name__=="__main__":
    # Create Inspiration
    Ny, Nx = 12, 12
    inspiration = np.full((Ny,Nx), KEY['space'])    # This is our world
    
    # Create a basic house
    H = [[KEY['wall'],KEY['wall'], KEY['wall'], KEY['wall']],
         [KEY['door'],KEY['space'],KEY['space'],KEY['wall']],
         [KEY['wall'],KEY['torch'],KEY['space'],KEY['wall']],
         [KEY['wall'],KEY['wall'], KEY['wall'], KEY['wall']]]
    H = np.array(H)
    
    # Place the house in the world
    inspiration[Ny/2:Ny/2+H.shape[0], Nx/2:Nx/2+H.shape[1]] = H
    
    # Create Style
    style = np.full((Ny,Nx), KEY['space'])    # This is our world
    
    # Create a basic house
    H = [[KEY['space'],KEY['wall'], KEY['wall'], KEY['space']],
         [KEY['wall'], KEY['wall'], KEY['wall'], KEY['wall']],
         [KEY['wall'], KEY['space'],KEY['torch'],KEY['wall']],
         [KEY['wall'], KEY['space'],KEY['space'],KEY['wall']],
         [KEY['wall'], KEY['space'],KEY['space'],KEY['wall']],
         [KEY['door'], KEY['torch'],KEY['space'],KEY['wall']],
         [KEY['wall'], KEY['wall'], KEY['wall'], KEY['wall']],
         [KEY['space'],KEY['wall'], KEY['wall'], KEY['space']]]
    H = np.array(H)
    
    # Place the house in the world
    style[2:10, 4:8] = H
    
    scores = [simple_score(inspiration), simple_score(style)]
    worlds = [inspiration, style]
    ScoreRegression(worlds, scores)
    #print(style)
    #print(inspiration)
    
    #print(FindAnalogy(inspiration, style)[0,:,:,0])