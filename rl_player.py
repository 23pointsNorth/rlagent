import math
import cv2
import numpy as np
import os

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
        self.gamma = 0.1#0.98
        self.learning_rate = 0.1
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

    def act(self, state):
        aprob = self.model.predict(state, batch_size=1).flatten()
        self.probs.append(aprob)
        prob = aprob / np.sum(aprob)
        action = np.random.choice(self.action_size, 1, p=prob)[0]
        if (np.random.rand() < 0.02):
            action = np.random.randint(0, self.action_size)
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

    def prepare_training(self):
        self.games += 1
        if (self.rewards[-1] > 0):
            self.wins += 1
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        rewards = self.discount_rewards(rewards)
        div_rew = np.std(rewards - np.mean(rewards))
        if (div_rew == 0):
            div_rew = np.finfo(np.float32).eps
        rewards = rewards / div_rew
        gradients *= rewards
        
        self.Xrmemory += self.states_roi
        self.Xamemory += self.states_a
        self.Ymemory = np.vstack((self.Ymemory, (self.probs + self.learning_rate * np.squeeze(np.vstack([gradients])))))
        self.states_roi, self.states_a, self.probs, self.gradients, self.rewards = [], [], [], [], []

    def prepare_training2(self):
        self.games += 1
        if (self.rewards[-1] > 0):
            self.wins += 1
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        rewards = self.discount_rewards(rewards)
        # rewards -= np.mean(rewards)
        div_rew = np.std(rewards - np.mean(rewards))
        if (div_rew == 0):
            div_rew = 1e-4 #np.finfo(np.float32).eps
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

        print 'Updating on ', Y.shape, ' items...'
        # self.model.train_on_batch(X, Y)
        self.model.fit(X, Y, batch_size=5000, epochs=self.training_epochs, verbose=1)
        print 'Model updated on last batch!'
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

if __name__ == '__main__':
    agent = PGAgent() #rll.create_model('model.h5')

    epoch = 0
    wins = 0
    score_sum = 0
    allowed_moves = 100
    wolrd_size = 250
    main_terrain_map = m.draw_terrain(shape=(wolrd_size, wolrd_size))
    goal_offset_size = 40
    main_tree_map = m.draw_trees(terrain=main_terrain_map, R=np.random.randint(3, 10), freq=np.random.randint(15, 35))
    # terrain_map = np.maximum(main_terrain_map, main_terrain_map) #main_terrain_map#
    terrain_map = np.maximum(main_tree_map, main_terrain_map) #main_terrain_map#

    # Clear the file
    with open(SCORES_FILE, 'w') as scores_file:
        scores_file.write('ratio,score,completeness_ratio\n')

    while True:
        epoch += 1
        print '----'
        print 'ID:', str(ID), ' Epoch: ', epoch, ' w: ', wins

        f_map = m.get_pretty_map(main_terrain_map, main_tree_map)
        # f_map = get_pretty_map(main_terrain_map, main_tree_map)

        grid = astar.SquareGrid(terrain_map)

        # Choose a good start and end point
        start = grid.choose_free_loc()
        goal = start
        while grid.dist(start, goal) <= 1.0:
            # After 50k epochs, make it harder
            if (epoch > 1e5):
                goal_offset_size = 64
                allowed_moves = 400
            goal_offset = (2*np.random.random(2) - 1) * (epoch % goal_offset_size + 3)
            goal = np.asarray(start) + goal_offset.astype(np.int)
            goal[goal < 0] = 0
            goal[goal >= wolrd_size] = wolrd_size - 1
            goal = tuple(goal)

        cv2.circle(f_map, start, 4, (120, 110, 250), -1)
        cv2.circle(f_map, goal, 4, (250, 110, 120), -1)

        total_r_states = []
        total_a_states = []
        total_actions = []

        score = 0
        last_location = None
        total_played_actions = 0
        fails = 0
        last_failed = False
        my_loc = start
        while True:
            roi = m.get_roi(my_loc, terrain_map, half_size=16)
            #cv2.imshow('roi', roi)
            r = np.asarray([[roi]], dtype='float32').reshape(1,32,32,1)/255.0
            a = np.asarray([math.atan2(my_loc[1] - goal[1], my_loc[0] - goal[0])])/math.pi
            action, prob = agent.act([r, a])
            total_played_actions += 1

            if "DISPLAY" in os.environ:
                cv2.imshow('roi2', (r[0]*255.0).astype(np.uint8))
            # print a

            # print 'Action: ', action, ' agent movement: ', offsets[action]
            my_new_loc = map(sub, my_loc, offsets[action])
            if (grid.in_bounds(my_new_loc) and grid.passable(my_new_loc)):
                # Possible move

                # Get cost of going to new locaiton
                #reward = - grid.dist(my_new_loc, goal) - grid.cost(my_loc, my_new_loc, last=last_location)
                reward = -1
                if (grid.dist(my_new_loc, goal) < grid.dist(my_loc, goal)):
                    reward = +1
                # Extra terrain rewards
                reward -= (grid.uphill_cost(my_loc, my_new_loc) / grid.UPHILL_COST_NORM)
                #reward = -1
                if (grid.dist(my_new_loc, goal) <= 1.):
                    reward += 10 #1e4
                if total_played_actions >= allowed_moves:
                    reward -= 10 #1e3
                score += reward

                # Remember transition
                agent.remember([r, a], action, prob, reward)

                # Update locations
                last_location = my_loc
                my_loc = my_new_loc

                # Draw my new location
                f_map[my_loc[1], my_loc[0], :] = [230, 20, 125]
                # cv2.circle(f_map, tuple(my_loc), 1, (128, 10, 190), -1)

                dist = grid.dist(my_loc, goal)
                
                if (dist <= 1.):
                    wins += 1
                    print 'Distance to goal: ', dist
                    print 'Reached goal with ', fails, ' hits'

                    # Train
                    agent.prepare_training2()
                    completeness_ratio = max(0, 1 - (grid.dist(my_loc, goal) / max(min(grid.dist(start, goal),grid.dist(my_loc, start)), 1e-4)))
                    agent.add_completeness_ratio(completeness_ratio)
                    # agent.train()
                    break # break
            else:
                # Impossible move 
                # agent.forget_last_action()
                # Remember transition
                reward = -10#0
                agent.remember([r, a], action, prob, reward)
                fails += 1

            if total_played_actions >= allowed_moves:
                # Train model again
                agent.prepare_training2()
                completeness_ratio = max(0, 1 - (grid.dist(my_loc, goal) / max(min(grid.dist(start, goal),grid.dist(my_loc, start)), 1e-4)))
                agent.add_completeness_ratio(completeness_ratio)
                # agent.train()
                print 'Cannot solve the current configuration... dist: ', '%.2f' % grid.dist(my_new_loc, goal)
                break

        print 'Finished epoch with ', total_played_actions, ' steps and score of ', score, ' ratio of: ', agent.get_ratio()
        score_sum += score
        # cv2.imshow('map', terrain_map)
        if "DISPLAY" in os.environ:
            cv2.imshow('path', f_map)
            cv2.waitKey(1)
        else:
            cv2.imwrite('path'+str(epoch%2)+'_'+str(ID)+'.jpg', f_map)

        if epoch > 1 and epoch % agent.traj_epochs == 0:
            print '>>> TRAINING !'

            with open(SCORES_FILE, 'a') as score_file:
                score_file.write('%.3f, %.2f, %.2f\n' % (agent.get_ratio(), float(score_sum)/agent.traj_epochs, agent.get_completeness_ratio()))
            score_sum = 0

            agent.train()
            main_tree_map = m.draw_trees(terrain=main_terrain_map, R=np.random.randint(3, 10), freq=np.random.randint(15, 35))
            terrain_map = np.maximum(main_tree_map, main_terrain_map) #main_terrain_map#

        if epoch > 1 and epoch % 10000 == 0 and "DISPLAY" not in os.environ:
            print 'Saving model..'
            agent.save('models/model_rl'+ str(epoch)+'_'+str(ID)+'.h5')
