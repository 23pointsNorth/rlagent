import collections
import heapq
import math
import cv2
import numpy as np

import map as m
from map import get_roi, get_pretty_map

import h5py

class PriorityQueue:
    def __init__(self):
        self.elements = []
    
    def empty(self):
        return len(self.elements) == 0
    
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self):
        return heapq.heappop(self.elements)[1]

class SquareGrid:
    WALL = 255
    UPHILL_COST = 20.0
    UPHILL_COST_EXP = 1.2
    TURN_COST = 02#10.0
    UPHILL_COST_NORM = UPHILL_COST * (255 ** UPHILL_COST_EXP)

    def __init__(self, grid):
        self.width = grid.shape[1]
        self.height = grid.shape[0]
        self.grid = grid
        # print '[A*] Grid shape: ', grid.shape
    
    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height
    
    def passable(self, id):
        (x, y) = id
        return self.grid[y, x] != self.WALL
    
    def neighbors(self, id):
        (x, y) = id
        results = [(x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1), # diagonals
                   (x+1, y), (x, y-1), (x-1, y), (x, y+1)] # up, down, left, right
                   
        # if (x + y) % 2 == 0: results.reverse() # aesthetics
        results = filter(self.in_bounds, results)
        results = filter(self.passable, results)
        return results
    
    def length(self, v):
        return math.sqrt(v[0]**2+v[1]**2)
    def dot_product(self, v, w):
       return v[0]*w[0] + v[1]*w[1]
    def determinant(self, v, w):
       return v[0]*w[1] - v[1]*w[0]
    def inner_angle(self, v, w):
        vl = self.length(v)
        wl = self.length(w)
        vlwl = vl * wl
        if (vlwl == 0):
            vlwl = np.finfo(np.float32).eps
        cosx = self.dot_product(v, w) / (vlwl)
        rad = math.acos(cosx) # in radians
        return rad * 180.0 / math.pi # returns degrees
    def angle_clockwise(self, A, B):
        inner=self.inner_angle(A,B)
        det = self.determinant(A,B)
        if det < 0: #this is a property of the det. If the det < 0 then B is clockwise of A
            return inner
        else: # if the det > 0 then A is immediately clockwise of B
            return 360-inner

    def cost(self, p0, p1, last=None): # from p0 to p1 cost
        turn_cost = 0
        if last:
            turn_cost = self.turn_cost(p0, p1, last)
            # print 'TURN_COST: ', turn_cost
        return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2) + turn_cost + \
               self.uphill_cost(p0, p1) # cost of going uphil p1 - p0

    def uphill_cost(self, p0, p1):
        return self.UPHILL_COST * (max(0, (int(self.grid[p1[1], p1[0]]) - int(self.grid[p0[1], p0[0]]))/1.0) ** self.UPHILL_COST_EXP) 
    def turn_cost(self, p0, p1, last):
        return abs(180-self.angle_clockwise([last[0] - p0[0], last[1] - p0[1]], [p1[0] - p0[0], p1[1] - p0[1]]))

    def dist(self, p0, p1):
        return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

    def choose_free_loc(self):
        loc = None
        while True:
            loc = (np.random.randint(0, self.width), np.random.randint(0, self.height))
            # print loc
            if self.passable(loc):
                break
        return loc

def heuristic(g, s, n): #goal, start, next
    CROSS_COST_W = 0.5#0.5
    # (x1, y1) = a
    # (x2, y2) = b
    # return abs(x1 - x2) + abs(y1 - y2)
    dx1 = n[1] - g[1]
    dy1 = n[0] - g[0]
    dx2 = s[1] - g[1]
    dy2 = s[0] - g[0]
    cross = abs(dx1*dy2 - dx2*dy1)
    return math.sqrt((n[0] - g[0])**2 + (n[1] - g[1])**2) + cross*CROSS_COST_W


def a_star_search(flight_map, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    last_loc = start


    # print flight_map.cost([10, 0], [9, 0], last=[9, 0])
    # return

    while not frontier.empty():
        
        current = frontier.get()
        if current == goal:
            break
        for next_loc in flight_map.neighbors(current):
            new_cost = cost_so_far[current] + flight_map.cost(current, next_loc, last=last_loc)
            if next_loc not in cost_so_far or new_cost < cost_so_far[next_loc]:
                cost_so_far[next_loc] = new_cost
                priority = new_cost + heuristic(goal, start, next_loc)
                frontier.put(next_loc, priority)
                came_from[next_loc] = current
        last_loc = current      
    return came_from, cost_so_far

def get_path(path_dict, goal):
    p = []
    if not (goal in path_dict):
        # didn't reach the goal
        print 'A* didn\'t find a goal'
        return p

    p.append(goal)
    c_id = goal
    while True:
        c_node = path_dict[c_id]
        if c_node == None:
            break
        else:
            p.append(c_node)
            c_id = c_node

    return p



if __name__ == '__main__':
    SHOW_PATH=True
    main_terrain_map = m.draw_terrain()
    main_tree_map = m.draw_trees(terrain=main_terrain_map)
    terrain_map = np.maximum(main_tree_map, main_terrain_map)
    trees_grid = SquareGrid(terrain_map)
    
    get_pretty_map(main_terrain_map, main_tree_map)
    roi_size = 32
    f = h5py.File('data.h5', 'w')
    img_dset = f.create_dataset('img', (0, roi_size, roi_size), dtype='i', chunks=True, maxshape=(None,roi_size, roi_size), compression='gzip', compression_opts=2)
    a_dset = f.create_dataset('angle', (0,), dtype='f', chunks=True, maxshape=(None,), compression='gzip', compression_opts=2)
    ga_dset = f.create_dataset('goal_angle', (0,), dtype='f', chunks=True, maxshape=(None,), compression='gzip', compression_opts=2)

    while True:
        # Choose start end locations
        start = trees_grid.choose_free_loc()
        goal = trees_grid.choose_free_loc()

        # start = (80, 10)
        # goal = (80, 290)

        came_from, cost_so_far = a_star_search(trees_grid, start, goal)
        print 'Done A*'

        l = get_path(came_from, goal)
        if (len(l)):
            terrain = get_pretty_map(main_terrain_map, main_tree_map)
            cv2.circle(terrain, start, 5, 80, -1)
            # print start

            rois = []
            angles = []
            goal_angles = []
            for i in xrange(len(l)-1):
                roi = get_roi(l[i], terrain_map, half_size=roi_size/2)
                angle = math.atan2(l[i][1] - l[i+1][1], l[i][0] - l[i+1][0])    
                rois.append(roi)
                angles.append(angle)
                goal_angles.append(math.atan2(l[i][1] - goal[1], l[i][0] - goal[0]))
            
            # Add to db
            img_dset.resize(len(img_dset) + len(rois), axis=0)
            img_dset[-len(rois):]=np.asarray(rois)
            a_dset.resize(len(a_dset) + len(angles), axis=0)
            a_dset[-len(angles):]=np.asarray(angles)
            ga_dset.resize(len(ga_dset) + len(goal_angles), axis=0)
            ga_dset[-len(goal_angles):]=np.asarray(goal_angles)

            print 'Dataset size: ', img_dset.shape
            if (img_dset.shape[0] >= 555000):
                print 'Reached 1e5 datapoints'
                break
            if SHOW_PATH:
                for i in l:
                    # (x, y) = i
                    # terrain[y, x]=128
                    cv2.circle(terrain, i, 1, 128, -1)
                    cv2.imshow("agent_view", get_roi(i, terrain))
                    cv2.imshow('Agent terrain map', get_roi(i, terrain_map))
                    cv2.waitKey(1)
                cv2.imshow("path", terrain)
            # print came_from
            #print ": ", l[::-1]
        print "Path length: ", len(l)
        key = cv2.waitKey(10)
        if (len(l)%10 == 0):
            # Change params
            print 'Changing parameters!'
            main_terrain_map = m.draw_terrain()
            main_tree_map = m.draw_trees(terrain=main_terrain_map, R=np.random.randint(1, 10), freq=np.random.randint(10, 100))
            terrain_map = np.maximum(main_tree_map, main_terrain_map)
            trees_grid = SquareGrid(terrain_map)


