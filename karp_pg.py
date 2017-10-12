""" implements a simple policy gradient (actor critic technically) agent """

import argparse
import time
import numpy as np
from scipy.signal import lfilter
from scipy.misc import imsave, imresize
import tensorflow as tf
import tensorflow.contrib.slim as slim
import cv2

import terrainEnv as te
import os

parser = argparse.ArgumentParser(description=None)
parser.add_argument('-e', '--env', default='Breakout-v3', type=str, help='gym environment')
parser.add_argument('-b', '--batch_size', default=10000, type=int, help='batch size to use during learning')
parser.add_argument('-l', '--learning_rate', default=1e-3, type=float, help='used for Adam')
parser.add_argument('-g', '--discount', default=0.99, type=float, help='reward discount rate to use')
parser.add_argument('-n', '--hidden_size', default=20, type=int, help='number of hidden units in net')
parser.add_argument('-c', '--gradient_clip', default=40.0, type=float, help='clip at this max norm of gradient')
parser.add_argument('-v', '--value_scale', default=0.5, type=float, help='scale of value function regression in loss')
parser.add_argument('-t', '--entropy_scale', default=0, type=float, help='scale of entropy penalty in loss')
parser.add_argument('-m', '--max_steps', default=1000000, type=int, help='max number of steps to run for')
args = parser.parse_args()
print(args)

device_instance = '/cpu:0'

# -----------------------------------------------------------------------------
# def process_frame(frame):
#     """ Atari specific preprocessing, consistent with DeepMind """
#     reshaped_screen = frame.astype(np.float32).mean(2)      # grayscale
#     resized_screen = imresize(reshaped_screen, (84, 110)) # downsample
#     x = resized_screen[18:102, :]                           # crop top/bottom
#     x = imresize(x, (42, 42)).astype(np.float32)                             # downsample
#     x *= (1.0 / 255.0)                                      # place in [0,1]
#     x = np.reshape(x, [42, 42, 1])                          # introduce channel
#     return x

def policy_spec(x):
  # self.state_in= tf.placeholder(shape=[None,s_size],dtype=tf.float32)
  # hidden = slim.fully_connected(self.state_in,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
  # self.output = slim.fully_connected(hidden,a_size,activation_fn=tf.nn.softmax,biases_initializer=None)
  
  net = slim.fully_connected(x, 32, activation_fn=tf.nn.elu, scope='fc1')
  net = slim.fully_connected(net, 32, activation_fn=tf.nn.elu, scope='fc2')
  # net = slim.flatten(net)
  action_logits = slim.fully_connected(net, num_actions, activation_fn=tf.nn.softmax, scope='fc_act')
  net_value = slim.fully_connected(net, 16, activation_fn=tf.nn.elu, scope='fc3_v')
  value_function = slim.fully_connected(net_value, 1, activation_fn=None, scope='fc_value')
  return action_logits, value_function

def rollout(n, max_steps_per_episode=5000):
  """ gather a single episode with current policy """
  observations, actions, rewards, discounted_rewards = [], [], [], []
  ob = env.reset()
  ep_steps = 0
  num_episodes = 0
  ep_start_pointer = 0
  prev_obf = None
  wins = 0
  # sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0, 'CPU': 4}, log_device_placement=False))
  # sess.run(tf.global_variables_initializer())
  while True:
    # print 'In while loop'
    # we concatenate the previous frame to get some motion information
    # obf_now = process_frame(ob)
    # obf_before = obf_now if prev_obf is None else prev_obf
    # obf = np.concatenate((obf_before, obf_now), axis=2)
    #obf = obf_now - obf_before
    # prev_obf = obf_now
    obf = ob

    # run the policy
    action = sess.run(action_index, feed_dict={x: np.expand_dims(obf, 0)}) # intro a batch dim
    action = action[0][0] # strip batch and #of samples from tf.multinomial
    # print 'after sess run'
    # execute the action
    ob, reward, done, info = env.step(action)
    # print 'after stepping env'
    # _, ob = obs
    ep_steps += 1

    observations.append(obf)
    actions.append(action)
    rewards.append(reward)

    if done or ep_steps >= max_steps_per_episode:
      # print 'Finished'
      cv2.imwrite('karp_pg_div1000.png', env.render(viz=False))
      if (rewards[-1] > 0):
        wins += 1
      num_episodes += 1
      ep_steps = 0
      prev_obf = None
      discounted_rewards.append(discount(rewards[ep_start_pointer:], args.discount))
      ep_start_pointer = len(rewards)
      ob = env.reset()
      if len(rewards) >= n: break

  return np.stack(observations), np.stack(actions), np.stack(rewards), np.concatenate(discounted_rewards), {'num_episodes':num_episodes, 'wins':wins}

# def rollout_log_result(main_list, result):
#   print 'callback: ', main_list.shape
#   main_list = np.append(main_list, np.asarray([result]), axis=0)
#   print main_list.shape

def env_resets(env):
  _, a = env.reset()
  return a

def env_step(x):
  env, action = x
  obs, reward, done, info = env.step(action)
  _, a = obs
  return [a, reward, done, info] 

import multiprocessing as mp
def parallel_rollout(envs, batch_size, threads = 4, max_steps_per_episode=5000):
  results = np.empty(shape=(len(envs), 0))
  env_done = np.zeros(shape=(len(envs), ))

  # observations, actions, rewards, discounted_rewards = np.empty(shape=(len(envs), 0)), np.empty(shape=(len(envs), 0)), np.empty(shape=(len(envs), 0)), np.empty(shape=(len(envs), ))
  observations = [ [] for _ in range(len(envs))]
  actions, rewards, discounted_rewards = [ [] for _ in range(len(envs))], [ [] for _ in range(len(envs))], [ [] for _ in range(len(envs))]
  # reset
  pool = mp.Pool(processes=threads)
  ob = pool.map(env_resets, envs)
  pool.close()
  pool.join()    
  print('initial obs: ' + str(ob))

  # Setup vars
  ep_steps = 0
  num_episodes = 0
  ep_start_pointer = 0
  prev_obf = None
  wins = 0

  # Main iteration loop
  while True:
    acts = sess.run(action_index, feed_dict={x: np.vstack(ob)}).flatten()
    print 'actions', acts

    pool2 = mp.Pool(processes=threads)
    res = pool2.map(env_step, zip(envs, acts))
    pool.close()
    pool.join()    
    print('env step: ' + str(res))

    # tack the results in diff arrays
    print len(envs)
    for i in range(len(envs)):
      print 'loop', i
      print 'obs: ', observations[i]#, observations[i].shape
      print 'res[i]:', res[i][0]
      # print 'appended: ', np.concatenate((observations[i], res[i][0]))
      observations[i].append(res[i][0])
      rewards[i].append(res[i][1])
      actions[i].append(actions[i])
      print 'appended after: ', observations[i]

    print 'all obs: ', observations 

    ep_steps += 1
    for id, re in enumerate(res):
      obs, reward, done, info = re
      if (done) or ep_steps >= max_steps_per_episode:
        env_done[id] = 1
        print 'done'
        if (rewards[-1] > 0):
          wins += 1
        num_episodes += 1
        ep_steps = 0
        prev_obf = None
        discounted_rewards[id] = discount(rewards, args.discount)

    print 'Discoutned rewatrds: ', discounted_rewards
    print np.asarray(discounted_rewards).shape
    # print rewards.shape
    # print actions.shape
    # break.  
  # pool = mp.Pool()
  # for i in range(threads):
  #   print 'Processing thread: ', i
  #   pool.apply_async(rollout,
  #                    args = (envs[i], float(batch_size)/threads, ), # each rollout will compute a small part
  #                    callback = lambda x: rollout_log_result(results, x))
  # pool.close()
  # pool.join()
  # result_list = np.asarray(result_list)
  # print 'Joined all'
  # Now compress list
  # samples = []
  # for c in result_list: # for each chain
  #   for s in c: # for each sample in the chain return
  #     samples.append(s)
  # samples = np.asarray(samples)

  # print 'Sample shape: ', samples.shape
  np.stack(results[0]), np.stack(results[1]), np.stack(results[2]), np.concatenate(results[3]), {'num_episodes':1, 'wins':1}

def discount(x, gamma): 
  return lfilter([1],[1,-gamma],x[::-1])[::-1]
# -----------------------------------------------------------------------------

# create the environment
env = te.TerrainEnv(angle_only=True, sparse_reward=True)
num_actions = env.action_space.n

#Store file 
ID = os.getenv('ID', 'not_added')
SCORES_FILE = 'scores'+str(ID)+'.csv'
# Clear the file
with open(SCORES_FILE, 'w') as scores_file:
  scores_file.write('ratio,score\n')
threads = 2
# envs = [te.TerrainEnv() for i in xrange(threads)]

# compile the model
x = tf.placeholder(tf.float32, (None, 1), name='x')
action_logits, value_function = policy_spec(x)
action_index = tf.multinomial(action_logits - tf.reduce_max(action_logits, 1, keep_dims=True), 1) # take 1 sample
# compile the loss: 1) the policy gradient
sampled_actions = tf.placeholder(tf.int32, (None,), name='sampled_actions')
discounted_reward = tf.placeholder(tf.float32, (None,), name='discounted_reward')
pg_loss = tf.reduce_mean((discounted_reward - value_function) * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=action_logits, labels=sampled_actions))
# and 2) the baseline (value function) regression piece
value_loss = args.value_scale * tf.reduce_mean(tf.square(discounted_reward - value_function))
# and 3) entropy regularization
action_log_prob = tf.nn.log_softmax(action_logits)
entropy_loss = -args.entropy_scale * tf.reduce_sum(action_log_prob*tf.exp(action_log_prob))
# add up and minimize
loss = pg_loss + value_loss + entropy_loss
# create the optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
grads = tf.gradients(loss, tf.trainable_variables())
grads, _ = tf.clip_by_global_norm(grads, args.gradient_clip) # gradient clipping
grads_and_vars = list(zip(grads, tf.trainable_variables()))
train_op = optimizer.apply_gradients(grads_and_vars)

# tf init
sess = tf.Session()
sess.run(tf.global_variables_initializer())
n = 0
mean_rewards = []
while n <= args.max_steps: # loop forever
  n += 1

  # collect a batch of data from rollouts and do forward/backward/update
  t0 = time.time()
  observations, actions, rewards, discounted_reward_np, info = rollout(args.batch_size)
  # observations, actions, rewards, discounted_reward_np, info = parallel_rollout(envs, args.batch_size, threads)
  t1 = time.time()
  sess.run(train_op, feed_dict={x:observations, sampled_actions:actions, discounted_reward:discounted_reward_np})
  t2 = time.time()

  average_reward = np.sum(rewards)/info['num_episodes']
  mean_rewards.append(average_reward)
  print('[ID: %s] step %d: collected %d frames in %.2fs, mean episode reward = %.2f (%d/%d (%.2f) eps), update in %.2fs' % (str(ID), n, observations.shape[0], t1-t0, average_reward, info['wins'], info['num_episodes'], float(info['wins'])/info['num_episodes'], t2-t1))
  # Save to file
  with open(SCORES_FILE, 'a') as score_file:
    score_file.write('%.3f, %.2f\n' % (float(info['wins'])/info['num_episodes'], average_reward))

print(args)
print('total average reward: %f +/- %f (min %f, max %f)' % \
      (np.mean(mean_rewards), np.std(mean_rewards), np.min(mean_rewards), np.max(mean_rewards)))
