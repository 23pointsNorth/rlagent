import math
import cv2
import numpy as np
import os

import terrainEnv as te
import map as m
# from map import get_roi, get_pretty_map
import astar as astar
import h5py
from operator import add, sub

import rl_learner as rll
import keras
from keras.utils import to_categorical
from keras.optimizers import SGD, Adam, Adamax, Adagrad

offsets = [[+1, 0], [+1, +1], [0, +1], [-1, +1], [-1, 0], [-1, -1], [0, -1], [+1, -1], [+1, 0]] # x,y

class PGAgent:
    '''
    Based on https://github.com/keon/policy-gradient
    '''
    def __init__(self):
        self.action_size = rll.num_classes
        self.gamma = 0.99
        self.learning_rate = 0.1
        self.learning_rate_step_down = 0.05
        self.traj_epochs = 40
        self.training_epochs = 2
        self.games = 0
        self.wins = 0
        self.completeness_ratio = []
        self.states_roi = []
        self.states_a = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.Xrmemory = []#np.empty(shape=(0,rll.img_rows, rll.img_cols, 1))
        self.Xamemory = []#np.empty(shape=(0,))
        self.Ymemory = np.empty(shape=(0, rll.num_classes))
        self.model = rll.create_model() # 'model.h5'
        self._config_model()
        self.model.summary()

    def _config_model(self):
        optimizer = Adagrad(lr=0.002) #.002self.learning_rate)
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=optimizer, 
                      metrics=['accuracy'] )

    def remember(self, state, action, prob, reward):
        y = np.zeros([self.action_size])
        y[action] = 1
        self.gradients.append(np.array(y).astype('float32') - prob)
        state_roi, state_a = state
        self.states_roi.append(state_roi)
        self.states_a.append(state_a)
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

    def forget_last_action(self):
        self.probs = self.probs[:-1]

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            # if rewards[t] != 0: # TEST
            #     running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # def prepare_training(self):
    #     self.games += 1
    #     if (self.rewards[-1] > 0):
    #         self.wins += 1
    #     gradients = np.vstack(self.gradients)
    #     rewards = np.vstack(self.rewards)
    #     rewards = self.discount_rewards(rewards)
    #     div_rew = np.std(rewards - np.mean(rewards))
    #     if (div_rew == 0):
    #         div_rew = np.finfo(np.float32).eps
    #     rewards = rewards / div_rew
    #     gradients *= rewards
        
    #     self.Xrmemory += self.states_roi
    #     self.Xamemory += self.states_a
    #     self.Ymemory = np.vstack((self.Ymemory, (self.probs + self.learning_rate * np.squeeze(np.vstack([gradients])))))
    #     self.states_roi, self.states_a, self.probs, self.gradients, self.rewards = [], [], [], [], []

    def prepare_training_norm(self):
        self.games += 1
        if (self.rewards[-1] > 1): #This assumes last reward is strictly > 1
            self.wins += 1
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        rewards = self.discount_rewards(rewards)
        div_rew = np.std(rewards - np.mean(rewards))
        if (div_rew == 0):
            div_rew = 1e-4
        rewards = rewards / div_rew
        gradients *= rewards
        
        self.Xrmemory += self.states_roi
        self.Xamemory += self.states_a
        self.Ymemory = np.vstack((self.Ymemory, (self.probs + self.learning_rate * np.squeeze(np.vstack([gradients])))))
        self.states_roi, self.states_a, self.probs, self.gradients, self.rewards = [], [], [], [], []

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
        # self.model.train_on_batch(X, Y)
        self.model.fit(X, Y, batch_size=5000, epochs=self.training_epochs, verbose=1)
        print '[PGAgent] ' + 'Model updated on last batch!'
        self.states_roi, self.states_a, self.probs, self.gradients, self.rewards = [], [], [], [], []
        self.Xrmemory, self.Xamemory = [], []
        self.Ymemory = np.empty(shape=(0, rll.num_classes))
        self.games, self.wins = 0, 0
        self.completeness_ratio = []

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
        # Now memorize
        # print 'Memorizing ', len(total_actions), ' cases.'
        # model.fit([np.asarray(total_r_states), np.asarray(total_a_states)], 
        #           to_categorical(total_actions, num_classes=rll.num_classes),
        #           batch_size=1, epochs=1, verbose=1)

ID = os.getenv('ID', -109)
SCORES_FILE = 'scores'+str(ID)+'.csv'

def agent_eval(env, agent):
    test_games = 30
    wins = 0
    for _ in xrange(test_games):
        r,a = env.reset()
        while True:
            action, _ = agent.act([r, a], eval_test=True)
            full_state, reward, is_done, _ = env.step(action)
            r, a = full_state
            if is_done:
                if reward > 1:
                    wins += 1
                break
    return float(wins)/test_games

if __name__ == '__main__':
    # Hyper params
    epoch = 0
    total_wins = 0
    score_sum = 0
    world_size = 250
    success_ratio = 0

    agent = PGAgent()
    env = te.TerrainEnv(world_size=world_size)

    # Clear the file
    with open(SCORES_FILE, 'w') as scores_file:
        scores_file.write('ratio,score,completeness_ratio,testing\n')

    while True:
        epoch += 1
        print '----'
        print 'ID:', str(ID), ' Epoch: ', epoch, ' w: ', total_wins + agent.wins

        score = 0
        fails = 0

        r, a = env.reset()
        while True:
            action, prob = agent.act([r, a])

            # Make the action
            full_state, reward, is_done, info = env.step(action)
            agent.remember([r, a], action, prob, reward)
            score += reward
            r, a = full_state

            if is_done:
                # Train model again
                agent.prepare_training_norm()
                agent.add_completeness_ratio(env.get_completeness_ratio())
                break

        print 'Finished epoch with ', env.total_played_actions, ' steps and score of ', score, ' ratio of: ', agent.get_ratio()
        score_sum += score

        should_viz = "DISPLAY" in os.environ
        f_map = env.render(viz=should_viz)
        if not should_viz:
            cv2.imwrite('path'+str(epoch%2)+'_'+str(ID)+'.png', f_map)

        if epoch > 1 and epoch % agent.traj_epochs == 0:
            print '>>> TRAINING !'

            with open(SCORES_FILE, 'a') as score_file:
                score_file.write('%.3f, %.2f, %.2f, ' % (agent.get_ratio(), float(score_sum)/agent.traj_epochs, agent.get_completeness_ratio()))
            score_sum = 0
            total_wins += agent.wins
            if (agent.traj_epochs * 500 < epoch):
                agent.learning_rate = agent.learning_rate_step_down
            agent.train()

            if epoch % (5 * agent.traj_epochs) == 0:
                print '>>> Evaluation at ', epoch
                success_ratio = agent_eval(env, agent)
                print 'Test argmax: ', success_ratio
            with open(SCORES_FILE, 'a') as score_file:
                score_file.write('%.2f\n' % (success_ratio))

        if epoch > 1 and epoch % 10000 == 0 and "DISPLAY" not in os.environ:
            print 'Saving model..'
            agent.save('models/model_rl'+ str(epoch)+'_'+str(ID)+'.h5')

