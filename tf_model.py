
import tensorflow as tf
import functools
from tensorflow.examples.tutorials.mnist import input_data

import terrainEnv as te

import os
from scipy.signal import lfilter
import numpy as np

def doublewrap(function):
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator

class Model:

    def __init__(self, image, keep_prob, goal_orient, label):
        self.image = image
        self.label = label
        self.keep_prob = keep_prob
        self.goal_orient = goal_orient
        self.prediction
        self.optimize
        self.error

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def prediction(self):
        x = self.image
        x_image = tf.reshape(x, [-1, 32, 32, 1])
            # First conv layer    
        W_conv1 = weight_variable([5, 5, 1, 64])
        b_conv1 = bias_variable([64])

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        # Maxpool
        h_pool1 = max_pool_2x2(h_conv1)

        # Second layer
        W_conv2 = weight_variable([5, 5, 64, 32])
        b_conv2 = bias_variable([32])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        # Maxpool
        h_pool2 = max_pool_2x2(h_conv2)

        # Third layer
        W_conv3 = weight_variable([5, 5, 32, 16])
        b_conv3 = bias_variable([16])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        # Maxpool
        h_pool3 = max_pool_2x2(h_conv3)

        # # Dense connected
        hln = 7 #128
        W_fc1 = weight_variable([4 * 4 * 16, hln]) 
        b_fc1 = bias_variable([hln])

        h_pool2_flat = tf.reshape(h_pool3, [-1, 4 * 4 * 16])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        concated = tf.concat([h_fc1, self.goal_orient], 1)
        # concated = self.goal_orient
        W_fc2 = weight_variable([hln + 1, 32])
        b_fc2 = bias_variable([32])
        h_fc2 = tf.nn.relu(tf.matmul(concated, W_fc2) + b_fc2)
        # Dropout
        # keep_prob = tf.placeholder(tf.float32)
        # h_fc2_drop = tf.nn.dropout(h_fc2, self.keep_prob)

        # Readout
        W_fc3 = weight_variable([32, 8]) #tf.shape(self.label)[1]
        b_fc3 = bias_variable([8])

        y_conv = tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3)

        # Value function
        W_fc_value = weight_variable([32, 1]) #tf.shape(self.label)[1]
        b_fc_value = bias_variable([1])
        y_value = (tf.matmul(h_fc2, W_fc_value) + b_fc_value)
        return y_conv, y_value

    @define_scope
    def optimize(self):
        # logprob = tf.log(self.prediction + 1e-12)
        # cross_entropy = -tf.reduce_sum(self.label * logprob)
        action_logits,_ = self.prediction
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=action_logits))
        optimizer = tf.train.AdamOptimizer(1e-4)
        return optimizer.minimize(cross_entropy)

    @define_scope
    def error(self):
        action_logits,_ = self.prediction
        mistakes = tf.not_equal(
            tf.argmax(self.label, 1), tf.argmax(action_logits, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def main1():
    mnist = input_data.read_data_sets('./mnist/', one_hot=True)
    image = tf.placeholder(tf.float32, [None, 784])
    label = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)
    goal_orient = tf.placeholder(tf.float32, [None, 1])
    model = Model(image, keep_prob, goal_orient, label)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    import numpy as np
    orient = np.zeros(shape=(10000, 1))
    print 'Beginning of training!'
    for _ in range(10):
      images, labels = mnist.test.images, mnist.test.labels
      error = sess.run(model.error, {image: images, label: labels, goal_orient: orient, keep_prob: 1.0})
      print('Test error {:6.2f}%'.format(100 * error))
      for _ in range(60):
        images, labels = mnist.train.next_batch(100)
        sess.run(model.optimize, {image: images, label: labels, goal_orient: orient[:100,:], keep_prob: 0.5})

def discount(x, gamma=0.99): 
  return lfilter([1],[1,-gamma],x[::-1])[::-1]

class agent():
    def __init__(self, lr, state_size, action_size): # state_size=784, action_size=8, h_size = 8
        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.x = tf.placeholder(tf.float32, [None, state_size])
        self.image = tf.reshape(self.x, [-1, 32, 32, 1])
        self.label = tf.placeholder(tf.float32, [None, action_size])
        self.keep_prob = tf.placeholder(tf.float32)
        self.goal_orient = tf.placeholder(tf.float32, [None, 1])
        self.action_logits, self.value_function = Model(self.image, self.keep_prob, self.goal_orient, self.label).prediction
        
        # self.state_in= tf.placeholder(shape=[None,s_size],dtype=tf.float32)
        # hidden = slim.fully_connected(self.state_in,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        # self.output = slim.fully_connected(hidden, action_size, activation_fn=tf.nn.softmax,biases_initializer=None)
        self.chosen_action = tf.argmax(self.action_logits, 1)

        #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.
        # self.discounted_reward = tf.placeholder(shape=[None],dtype=tf.float32)
        self.discounted_reward = tf.placeholder(tf.float32, (None,), name='discounted_reward')
        self.sampled_actions = tf.placeholder(tf.int32, (None,), name='sampled_actions')
        
        # self.indexes = tf.range(0, tf.shape(self.action_logits)[0]) * tf.shape(self.action_logits)[1] + self.sampled_actions
        # self.responsible_outputs = tf.gather(tf.reshape(self.action_logits, [-1]), self.indexes) # Gather the flattened indexes

        # Compute loss
        # PG loss
        pg_loss = tf.reduce_mean((self.discounted_reward - self.value_function) * \
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.action_logits, labels=self.sampled_actions))
        # Baseline value function regression
        value_scale = 0.5
        value_loss = value_scale * tf.reduce_mean(tf.square(self.discounted_reward - self.value_function))
        # entropy regularization
        action_log_prob = tf.nn.log_softmax(self.action_logits)
        entropy_scale = 0.001
        entropy_loss = -entropy_scale * tf.reduce_sum(action_log_prob*tf.exp(action_log_prob))
        self.loss = pg_loss + value_loss + entropy_loss 
        
        #-tf.reduce_mean(tf.log(self.responsible_outputs)*self.discounted_reward)
        
        # Compute gradients
        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx, var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)
        
        self.gradients = tf.gradients(self.loss, tvars)
        gradient_clip = 40
        self.gradients, _ = tf.clip_by_global_norm(self.gradients, gradient_clip) # gradient clipping

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))

def policy_rollout():
    pass

def main():
    env = te.TerrainEnv()
    tf.reset_default_graph() # Clear the Tensorflow graph.

    #Store file 
    ID = os.getenv('ID', 'not_added')
    SCORES_FILE = 'scores'+str(ID)+'.csv'
    # Clear the file
    with open(SCORES_FILE, 'w') as scores_file:
      scores_file.write('ratio,score\n')

    myAgent = agent(lr=1e-4, state_size=32*32, action_size=env.action_space.n) #Load the agent.

    total_episodes = 500000 # Set total number of episodes to train agent on.
    max_ep = 400
    update_frequency = 20

    init = tf.global_variables_initializer()

    # Launch the tensorflow graph
    with tf.Session() as sess:
        sess.run(init)
        i = 0
        total_reward = []
        # total_length = []
        wins = 0
        num_episodes = 0
        reward_epoch_hist = []
            
        gradBuffer = sess.run(tf.trainable_variables())
        for ix, grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0
            
        while i < total_episodes:
            img, gorient = env.reset()
            # print ' >>> Reset. Episode: ', i 

            running_reward = 0
            
            ep_history = []
            for j in range(max_ep):
                # print j
                # print np.asarray([img])
                # print np.asarray(img).shape
                # print 'asd', gorient
                # Probabilistically pick an action given our network outputs.
                a_dist = sess.run(myAgent.action_logits, feed_dict={myAgent.image: img, myAgent.goal_orient: [gorient], myAgent.keep_prob: 1.0})
                if (i%200 ==0):
                    print a_dist, gorient
                a = np.random.choice(a_dist[0], p=a_dist[0])
                a = np.argmax(a_dist == a) #[0 0 0 a 0 0] -> max -> index(a)
                # print 'Action: ', a
                s1, r, d, info = env.step(a) #Get our reward for taking an action
                img1, gorient1 = s1
                ep_history.append([img, gorient, a, r, img1, gorient1])
                img = img1
                gorient = gorient1
                running_reward += r


                if d == True:
                    if (i%50 == 0):
                        env.render()
                    if (r > 0): # If last reward positive
                        wins += 1
                    num_episodes += 1
                    reward_epoch_hist.append(running_reward)
                    #Update the network.
                    ep_history = np.array(ep_history)
                    ep_history[:,3] = discount(ep_history[:,3], gamma=0) #discount reward
                    # ep_history[:,3] -= np.mean(ep_history[:,3])
                    std_div = np.std(ep_history[:,3] - np.mean(ep_history[:,3]))
                    if (std_div == 0):
                        std_div = 1e-4
                    ep_history[:,3] /= std_div 
                    # print 'Episode end. Score: ', running_reward, ep_history[:,0].shape

                    feed_dict={myAgent.discounted_reward: ep_history[:,3],
                               myAgent.sampled_actions: ep_history[:,2],
                               myAgent.image: np.vstack(ep_history[:,0]),
                               myAgent.goal_orient: np.vstack(ep_history[:,1]),
                               myAgent.keep_prob: 1.0}
                    grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
                    for idx,grad in enumerate(grads):
                        gradBuffer[idx] += grad

                    if i % update_frequency == 0 and i != 0:
                        print ' >>> >>> Training the network'
                        feed_dict = dict(zip(myAgent.gradient_holders, gradBuffer))
                        _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
                        for ix, grad in enumerate(gradBuffer):
                            gradBuffer[ix] = grad * 0

                        # Print statistics
                        avg_reward = np.sum(reward_epoch_hist)/num_episodes
                        print('[ID: %s] step %d: mean episode reward = %.2f (%d/%d (%.2f) eps)' % (str(ID), i, avg_reward, wins, num_episodes, float(wins)/num_episodes))
                        # Save to file
                        with open(SCORES_FILE, 'a') as score_file:
                            score_file.write('%.3f, %.2f\n' % (float(wins)/num_episodes, avg_reward))
                        # Reset vars
                        wins = 0
                        num_episodes = 0
                        reward_epoch_hist = []

                    
                    total_reward.append(running_reward)
                    # total_length.append(j)
                    break
                # print 'Running reward: ', running_reward
            
                #Update our running tally of scores.
            if i % 100 == 0 and i > 0:
                print('Mean total reward: ', np.mean(total_reward[-100:]))
            i += 1

def main3():
    # Setup environment 
    env = te.TerrainEnv()
    env.reset()

    # Params
    learning_rate = 1e-2
    gamma = 0.99 # discount factor for reward

    # Model
    tf.reset_default_graph()
    image = tf.placeholder(tf.float32, [None, 784])
    label = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)
    goal_orient = tf.placeholder(tf.float32, [None, 1])
    probability = Model(image, keep_prob, goal_orient, label)

    # From here we define the parts of the network needed for learning a good policy.
    tvars = tf.trainable_variables()
    input_y = tf.placeholder(tf.float32, [None,1], name="input_y")
    advantages = tf.placeholder(tf.float32, name="reward_signal")

    # The loss function. This sends the weights in the direction of making actions 
    # that gave good advantage (reward over time) more likely, and actions that didn't less likely.
    loglik = tf.log(input_y*(input_y - probability) + (1 - input_y)*(input_y + probability))
    loss = -tf.reduce_mean(loglik * advantages) 
    newGrads = tf.gradients(loss,tvars)

    # Once we have collected a series of gradients from multiple episodes, we apply them.
    # We don't just apply gradeients after every episode in order to account for noise in the reward signal.
    adam = tf.train.AdamOptimizer(learning_rate=learning_rate) # Our optimizer
    W1Grad = tf.placeholder(tf.float32, name="batch_grad1") # Placeholders to send the final gradients through when we update.
    W2Grad = tf.placeholder(tf.float32, name="batch_grad2")
    batchGrad = [W1Grad,W2Grad]
    updateGrads = adam.apply_gradients(zip(batchGrad, tvars))

    
    xs,hs,dlogps,drs,ys,tfps = [],[],[],[],[],[]
    running_reward = None
    reward_sum = 0
    episode_number = 1
    total_episodes = 10000
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        rendering = False
        sess.run(init)
        observation = env.reset() # Obtain an initial observation of the environment

        # Reset the gradient placeholder. We will collect gradients in 
        # gradBuffer until we are ready to update our policy network. 
        gradBuffer = sess.run(tvars)
        for ix,grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0
        
        while episode_number <= total_episodes:
            
            # Rendering the environment slows things down, 
            # so let's only look at it once our agent is doing a good job.
            if reward_sum/batch_size > 100 or rendering == True : 
                env.render()
                rendering = True
                
            # Make sure the observation is in a shape the network can handle.
            x = np.reshape(observation, [1,D])
            
            # Run the policy network and get an action to take. 
            tfprob = sess.run(probability,feed_dict={observations: x})
            action = 1 if np.random.uniform() < tfprob else 0
            
            xs.append(x) # observation
            y = 1 if action == 0 else 0 # a "fake label"
            ys.append(y)

            # step the environment and get new measurements
            observation, reward, done, info = env.step(action)
            reward_sum += reward

            drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

            if done: 
                episode_number += 1
                # stack together all inputs, hidden states, action gradients, and rewards for this episode
                epx = np.vstack(xs)
                epy = np.vstack(ys)
                epr = np.vstack(drs)
                tfp = tfps
                xs,hs,dlogps,drs,ys,tfps = [],[],[],[],[],[] # reset array memory

                # compute the discounted reward backwards through time
                discounted_epr = discount_rewards(epr)
                # size the rewards to be unit normal (helps control the gradient estimator variance)
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr //= np.std(discounted_epr)
                
                # Get the gradient for this episode, and save it in the gradBuffer
                tGrad = sess.run(newGrads,feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})
                for ix,grad in enumerate(tGrad):
                    gradBuffer[ix] += grad
                    
                # If we have completed enough episodes, then update the policy network with our gradients.
                if episode_number % batch_size == 0: 
                    sess.run(updateGrads,feed_dict={W1Grad: gradBuffer[0],W2Grad:gradBuffer[1]})
                    for ix,grad in enumerate(gradBuffer):
                        gradBuffer[ix] = grad * 0
                    
                    # Give a summary of how well our network is doing for each batch of episodes.
                    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                    print('Average reward for episode %f.  Total average reward %f.' % (reward_sum//batch_size, running_reward//batch_size))
                    
                    if reward_sum//batch_size > 200: 
                        print("Task solved in",episode_number,'episodes!')
                        break
                        
                    reward_sum = 0
                
                observation = env.reset()
            
    print(episode_number,'Episodes completed.')

if __name__ == '__main__':
  # main1()
  main()

