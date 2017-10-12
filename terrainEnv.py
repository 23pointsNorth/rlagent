
import astar
import map as m
import numpy as np
import cv2
from operator import add, sub
import math


class TerrainEnv:
    """docstring for TerrainEnv"""
    class ActionSpace:
        n = 8
        shape = (8,)
        low = 0 ### ???
        high = 7 ## ???
        offsets = [[+1, 0], [+1, +1], [0, +1], [-1, +1], [-1, 0], [-1, -1], [0, -1], [+1, -1], [+1, 0]] # x,y
        def sample():
            return np.random.randint(0, self.n)
    class ObservationSpace:
        shape = (1, )
        # shape = (32, 32)
    action_space = ActionSpace()
    observation_space = ObservationSpace()

    def __init__(self, world_size = 128, instance_name='TerrainEnv', obstacles=False, 
                 goal_hardness = 31, allowed_moves = 100, sparse_reward=False, angle_only=False,
                 env_cost=False):
        self.instance_name = instance_name
        self.world_size = world_size
        self.allowed_moves = allowed_moves
        self.obstacles = obstacles
        self.goal_hardness = goal_hardness
        self.min_distance = 3.0
        self.reset_counter = 0
        self.sparse_reward = sparse_reward
        self.angle_only = angle_only
        self.env_cost = env_cost

        # Move terrain and tree generation in reset if needed
        self.main_terrain_map = m.draw_terrain(shape=(self.world_size, self.world_size))
        self.reset()        
        
    def step(self, action):
        self.total_played_actions += 1
        my_new_loc = map(sub, self.my_loc, self.action_space.offsets[action])
        done = False
        if (self.grid.in_bounds(my_new_loc) and self.grid.passable(my_new_loc)):
            # Possible move
            info = 'Possible move ' + str(action)
            # Get cost of going to new location
            if (self.sparse_reward):
                reward = 0.0
            else:
                # reward = -self.grid.dist(my_new_loc, self.goal) - self.grid.cost(self.my_loc, my_new_loc, last=self.last_location)
                if (self.grid.dist(my_new_loc, self.goal) < self.grid.dist(self.my_loc, self.goal)):
                    reward = +1
                else:
                    reward = -1

                if (self.env_cost):
                    reward -= (self.grid.uphill_cost(self.my_loc, my_new_loc) / self.grid.UPHILL_COST_NORM)

            if (self.grid.dist(my_new_loc, self.goal) <= self.min_distance):
                info = 'Done'
                done = True
                if (self.sparse_reward):
                    reward += 1.0
                else:
                    reward += 10#1e4
                print '>>> >>> >>> [TerrainEnv]: ' + 'Solved map!'

            # Update locations
            self.last_location = self.my_loc
            self.my_loc = my_new_loc
        else:
            if (self.sparse_reward):
                reward = -1.0
            else:
                reward = -10
            done = True #####
            info = 'Impossible move ' + str(self.my_loc) + str(my_new_loc) + str(self.grid.in_bounds(my_new_loc)) #+ str(self.grid.passable(my_new_loc))

        if self.total_played_actions >= self.allowed_moves:
            info += ' Didn\'t reach goal'
            done = True
            if (self.sparse_reward):
                reward -=1.0
            else:
                reward -= 10 #1e3

        self.moves.append(self.my_loc)
        r, a = self.observe()
        if (self.angle_only):
            return a, reward, done, info
        else:
            return [r, a], reward, done, info

    def observe(self):
        roi = astar.get_roi(self.my_loc, self.terrain_map, half_size=16)
        # r = np.asarray([[roi]], dtype='float32').reshape(32,32,1)/255.0 # 1, 32, 32, 1
        r = np.asarray(roi, dtype='float32').reshape(1, 32, 32, 1)/255.0*0.0 # 1, 32, 32, 1
        # print 'r shape: ', r.shape
        a = np.asarray([math.atan2(self.my_loc[1] - self.goal[1], self.my_loc[0] - self.goal[0])])/math.pi
        return r, a

    def reset(self):
        self.reset_counter += 1
        self.main_tree_map = np.zeros(shape=self.main_terrain_map.shape)
        self.goal_offset_size = 32
        if (self.obstacles):
            self.main_tree_map = m.draw_trees(terrain=self.main_terrain_map, R=np.random.randint(4, 10), freq=np.random.randint(10, 20))
        self.terrain_map = np.maximum(self.main_tree_map, self.main_terrain_map)

        self.grid = astar.SquareGrid(self.terrain_map)
        # Choose a good start and end point
        self.start = self.grid.choose_free_loc()
        self.goal = self.start
        while self.grid.in_bounds(self.goal) and self.grid.passable(self.goal) and self.grid.dist(self.start, self.goal) <= self.min_distance:
            # After 50k epochs, make it harder
            # if (epoch > 5e4):
            #     self.goal_offset_size = 64
            #     allowed_moves = 400
            goal_offset = (2 * np.random.random(2) - 1) * ((self.goal_hardness + self.reset_counter) % self.goal_offset_size + self.min_distance + 1)
            self.goal = np.asarray(self.start) + goal_offset.astype(np.int)
            self.goal = tuple(self.goal)

        self.my_loc = self.start
        self.last_location = self.my_loc
        self.total_played_actions = 0
        self.moves = [self.my_loc]

        r, a = self.observe()
        if (self.angle_only):
            return a
        else:
            return r, a

    def render(self, viz=True):
        f_map = astar.get_pretty_map(self.main_terrain_map, self.main_tree_map)

        cv2.circle(f_map, self.start, 4, (120, 110, 250), -1)
        cv2.circle(f_map, self.goal, 4, (250, 110, 120), -1)

        # Draw list of moves/locations
        for l in self.moves:
            f_map[l[1], l[0], :] = [230, 20, 125]

        if (viz):
            cv2.imshow(self.instance_name, f_map)
            cv2.waitKey(1)
        return f_map


def main():
    print 'Illustrating random movements'
    env = TerrainEnv()
    print env.observation_space.shape
    print env.action_space.shape
    for _ in xrange(100):
        action = np.random.randint(0,9)
        observations, reward, done, info = env.step(action)
    env.render()
    a = env.reset()
    print len(a)

if __name__ == '__main__':
    main()