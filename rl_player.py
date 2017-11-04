import math
import cv2
import numpy as np
import os

from math import sqrt

import terrainEnv as te
import map as m
import astar as astar
import h5py
from operator import add, sub

import rl_learner as rll
import keras
from keras.utils import to_categorical
from keras.optimizers import SGD, Adam, Adamax, Adagrad

offsets = [[+1, 0], [+1, +1], [0, +1], [-1, +1], [-1, 0], [-1, -1], [0, -1], [+1, -1], [+1, 0]] # x,y 

class PGAgent:
    def __init__(self, model, action_size):
        self.action_size = action_size
        self.gamma = 0.97
        self.learning_rate_step_down = [0.1, 0.075, 0.05]
        self.learning_rate_step_down_epochs = [200, 400, 600, 100000]
        self.learning_rate_id = 0 
        self.traj_epochs = 40
        self.training_epochs = 2
        self.games = 0
        self.wins = 0
        self.completeness_ratio = []
        self.states_roi = []
        self.states_a = []
        self.gradients = []
        self.rewards = []
        self.insta_rewards = []
        self.probs = []
        self.Xrmemory = []
        self.Xamemory = []
        self.Ymemory = np.empty(shape=(0, self.action_size))
        self.model = model
        self._config_model()
        self.model.summary()

    def _config_model(self):
        optimizer = Adagrad(lr=0.002)
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=optimizer, metrics=['accuracy'] )

    def remember(self, state, action, prob, reward):
        y = np.zeros([self.action_size])
        y[action] = 1
        self.gradients.append(np.array(y).astype('float32') - prob)
        state_roi, state_a = state
        self.states_roi.append(state_roi)
        self.states_a.append(state_a)
        if (len(reward) > 1):
            self.rewards.append(reward[0])
            self.insta_rewards.append(reward[1])
        else:
            self.rewards.append(reward)

    def remember_simp(self, state, action, prob, reward):
        self.gradients.append(action - prob)
        state_roi, state_a = state
        self.states_roi.append(state_roi)
        self.states_a.append(state_a)
        if (len(reward) > 1):
            self.rewards.append(reward[0])
            self.insta_rewards.append(reward[1])
        else:
            self.rewards.append(reward)

    def act(self, state, eval_test=False):
        aprob = self.model.predict(state, batch_size=1).flatten()
        prob = aprob / np.sum(aprob)
        if (not eval_test):
            self.probs.append(aprob)
            action = np.random.choice(self.action_size, 1, p=prob)[0]
            if (np.random.rand() < 0.02):
                action = np.random.randint(0, self.action_size)
        else:
            action = np.argmax(prob)
        return action, prob

    def act_simp(self, state):
        mu = self.model.predict(state, batch_size=1).flatten()
        std = 0.05
        cause = np.random.normal(loc=mu, scale=std)
        cause[cause > 1.0] = 1.0
        cause[cause < 0] = 0
        self.probs.append(cause)
        return cause, mu

    def forget_last_action(self):
        self.probs = self.probs[:-1]

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
        running_add = 0.0
        for t in reversed(range(0, rewards.size)):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def prepare_training_norm(self, info=None):
        self.games += 1
        if (self.rewards[-1] > 1 or info is 'Done'):
            self.wins += 1
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        rewards = self.discount_rewards(rewards) + \
                        np.vstack(self.insta_rewards).astype(np.float32)
        rewards -= np.mean(rewards)
        div_rew = np.std(rewards)
        if (div_rew == 0):
            div_rew = 1e-4
        rewards = rewards / div_rew
        gradients *= rewards
        
        self.Xrmemory += self.states_roi
        self.Xamemory += self.states_a
        learning_rate = self.learning_rate_step_down[self.learning_rate_id]
        
        self.Ymemory = np.vstack((self.Ymemory, (self.probs + \
                        learning_rate * np.squeeze(np.vstack([gradients])))))
        self.states_roi, self.states_a, self.probs, self.gradients = [], [], [], []
        self.rewards, self.insta_rewards = [], []

    def get_ratio(self):
        return float(self.wins)/self.games

    def add_completeness_ratio(self, cr):
        self.completeness_ratio.append(cr)

    def get_completeness_ratio(self):
        return np.mean(self.completeness_ratio)

    def train(self):
        X = [np.vstack(self.Xrmemory), np.vstack(self.Xamemory)]
        Y = self.Ymemory
        Y[Y<0]=0.0
        Y[Y>1]=1.0

        print '[PGAgent] ' + 'Updating on ', Y.shape, ' items...'
        self.model.fit(X, Y, batch_size=5000, epochs=self.training_epochs, verbose=1)
        print '[PGAgent] ' + 'Model updated on last batch!'
        self.states_roi, self.states_a, self.probs, self.gradients, self.rewards = [], [], [], [], []
        self.Xrmemory, self.Xamemory = [], []
        self.Ymemory = np.empty(shape=(0, self.action_size))
        self.games, self.wins = 0, 0
        self.completeness_ratio = []

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


SIMP_ON = os.getenv('SIMP', False)
ID = os.getenv('ID', 'not_specified')
SCORES_FILE = 'scores' + str(ID) + '.csv'

def agent_eval(env, agent):
    test_games = 60
    wins = 0
    for i in xrange(test_games):
        r,a = env.reset(hard=True)
        while True:
            action, _ = agent.act([r, a], eval_test=True)
            full_state, reward, is_done, info = env.step(action)
            r, a = full_state
            if (len(reward) > 1):
                reward = reward[0]
            if is_done:
                if reward > 1 or info is 'Done':
                    wins += 1
                # Save state
                img = env.render(viz = False)
                cv2.imwrite('eval_traj_' + str(ID) + '_' + str(i) + '.png', img)
                break
    return float(wins) / test_games

if __name__ == '__main__':
    # Hyper params
    epoch = 0
    total_wins = 0
    score_sum = 0
    world_size = 250
    success_ratio = 0
    weight_ws = []
    scores_ws = []
    SIMP_REWARD_SCALE = 5.0

    agent = PGAgent(model = rll.create_model(), action_size = rll.num_classes)
    if SIMP_ON:
        print 'Initializing Simplification agent'
        simp_agent = PGAgent(model = rll.create_simp_model(), action_size = 32*32+1)
    env = te.TerrainEnv(world_size=world_size, obstacles=True, 
                        env_cost=True, sparse_reward=False,
                        env_cost_scale=50.0)

    # Clear the file
    with open(SCORES_FILE, 'w') as score_file:
        score_file.write('ratio,score,completeness_ratio,testing')
        if SIMP_ON:
            score_file.write(',weight_sum,weight_sum_std\n')
        else:
            score_file.write('\n')

    while True:
        epoch += 1
        print '----'
        print 'ID:', str(ID), ' Epoch: ', epoch, ' w: ', total_wins + agent.wins

        score = 0
        fails = 0

        r, a = env.reset()
        while True:
            w, mu = simp_agent.act_simp([r, a]) if SIMP_ON else (np.ones(32*32+1), np.ones(32*32+1))

            w[w < 0.5] = 0.0
            w[w != .0] = 1.0
            # print w, np.count_nonzero(w), np.mean(w)
            weight_ws.append(w)
            wr = (w[:-1].reshape((int(sqrt(len(w[:-1]))), -1)) * r.reshape((r.shape[1], -1))).reshape(r.shape)
            wa = w[-1] * a
            action, prob = agent.act([wr, wa])

            # Make the action
            full_state, reward, is_done, info = env.step(action)
            agent.remember([wr, wa], action, prob, reward)
            if (SIMP_ON):
                simp_reward = [reward[0], reward[1] - (np.mean(w)) * SIMP_REWARD_SCALE]
                simp_agent.remember_simp([r, a], w, mu, simp_reward)
            score += sum(reward)
            r, a = full_state

            if is_done:
                # Prepare gradients
                agent.prepare_training_norm(info)
                agent.add_completeness_ratio(env.get_completeness_ratio())
                if (SIMP_ON):
                    simp_agent.prepare_training_norm(info)
                break

        print 'Finished epoch with ', env.total_played_actions, ' steps and score of ', score, ' ratio of: ', agent.get_ratio()
        score_sum += score
        scores_ws.append(score)

        should_viz = "DISPLAY" in os.environ
        f_map = env.render(viz=should_viz)
        if not should_viz:
            cv2.imwrite('path' + str(epoch%2) + '_' + str(ID) + '.png', f_map)

        if epoch > 1 and epoch % agent.traj_epochs == 0:
            print '>>> TRAINING !'

            with open(SCORES_FILE, 'a') as score_file:
                score_file.write('%.3f, %.2f, %.2f, ' % (agent.get_ratio(), float(score_sum)/agent.traj_epochs, agent.get_completeness_ratio()))
            score_sum = 0
            total_wins += agent.wins

            lr_id = next(x[0] for x in enumerate(agent.learning_rate_step_down_epochs) if x[1] > epoch // agent.traj_epochs)
            lr_id = lr_id if lr_id < len(agent.learning_rate_step_down) else len(agent.learning_rate_step_down)-1
            agent.learning_rate_id = lr_id

            agent.train()
            if SIMP_ON:
                print 'Training Simplification Agent...'
                simp_agent.train()

            if epoch % (5 * agent.traj_epochs) == 0:
                print '>>> Evaluation at ', epoch
                success_ratio = agent_eval(env, agent)
                print 'Test argmax: ', success_ratio
            with open(SCORES_FILE, 'a') as score_file:
                score_file.write('%.2f' % (success_ratio))
                if SIMP_ON:
                    score_file.write(',%.2f,%.2f\n' % (np.mean(weight_ws), np.std(np.mean(weight_ws, axis=1))))
                else:
                    score_file.write('\n')
            weight_ws = []

        if epoch > 1 and epoch % 10000 == 0 and "DISPLAY" not in os.environ:
            print 'Saving model..'
            agent.save('models/model_rl'+ str(epoch)+'_'+str(ID)+'.h5')

