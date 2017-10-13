import numpy as np
# import math
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as colors

DEFAULT_HEIGHT, DEFAULT_WIDTH = (240, 320)

def draw_terrain(shape = [DEFAULT_HEIGHT, DEFAULT_WIDTH], mountains_scale=1.0, water_percentage=0.1):
    height, width = shape
    # scale = 1e-4
    height_map = np.zeros((height,width), np.float32)
    noise_scale = 1.0#0.8
    noise_freq_scale = 2.0 

    # Create map
    for y in xrange(height):
        for x in xrange(width):
            nx = float(x)/width - 0.5
            ny = float(y)/height - 0.5
            
            n = (noise_scale ** 0) * noise((2 * noise_freq_scale) * nx, (2 * noise_freq_scale) * ny) + \
                (noise_scale ** 1) * noise((2 * noise_freq_scale ** 1) * nx, (2 * noise_freq_scale ** 1) * ny) + \
                (noise_scale ** 2) * noise((2 * noise_freq_scale ** 2) * nx, (2 * noise_freq_scale ** 2) * ny)
            # print 'map.py : ' + n
            height_map[y, x] = (n) ** mountains_scale

    # Add water
    if (water_percentage != .0):
        sortedarr = height_map.copy().flatten()
        np.sort(sortedarr)
        thr = sortedarr[int(height_map.size * water_percentage)]
        height_map[height_map<thr] = thr

    view_map = height_map - height_map.min()
    view_map *= (255.0/view_map.max())

    return view_map.astype(np.uint8)


def draw_trees(terrain = None, water = True, freq = 20, R = 3, kernel_sz = 5):
    height, width = DEFAULT_HEIGHT, DEFAULT_WIDTH
    if (terrain is not None):
        height, width = terrain.shape
    # scale = 1e-4
    view_map = np.zeros((height,width), np.uint8)
    tree_map = np.zeros((height,width), np.uint8)

    min_v = 0
    if water and terrain is not None:
        min_v = terrain.min()

    # Create map
    for y in xrange(height):
        for x in xrange(width):
            # view_map[x,y]= int(255.0 * noise(y + centre[1]/scale, x + centre[0]/scale))
            view_map[y, x] = 255 * noise(freq * float(x)/width - 0.5, freq * float(y)/height - 0.5)
            if terrain is not None and terrain[y, x] == min_v:
                view_map[y, x] = 0

    # print view_map
    # return
    # Place trees
    for y in xrange(R, height-R):
        for x in xrange(R, width-R):
            maxv = view_map[y-R:y+R, x-R:x+R].max()
            # print view_map[y-R:y+R, x-R:x+R]

            if (view_map[y, x] == maxv and maxv != min_v):
                tree_map[y, x] = 255

    tree_map = cv2.dilate(tree_map, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_sz, kernel_sz)))
    # tree_map = cv2.morphologyEx(tree_map, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    # tree_map = cv2.dilate(tree_map, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    # cv2.imshow("Map", view_map)
    # cv2.imshow("Tree Map", tree_map)
    # cv2.waitKey(0)
    return tree_map

from opensimplex import OpenSimplex
gen = OpenSimplex()
def noise(nx, ny):
    # Rescale from -1.0:+1.0 to 0.0:1.0
    # print nx, ny
    return (gen.noise2d(nx, ny) / 2.0 + 0.5)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def get_pretty_map(terrain, trees, water=True):
    WATER_COLOUR = [200, 10, 10] # BGR
    TREE_COLOUR = [10, 255, 10]
    empty = np.zeros(terrain.shape, np.uint8)

    # Water
    water = (terrain == terrain.min()).astype(np.uint8)
    water_t = cv2.merge([water * WATER_COLOUR[i] for i in xrange(0,3)]) 

    # Trees
    trees_scaled = trees.astype(np.float) / 255
    tree_t = cv2.merge([trees_scaled * TREE_COLOUR[i] for i in xrange(0,3)]) 

    # Terrain elevation
    cmap = plt.get_cmap('terrain')
    ground_cmap = truncate_colormap(cmap, 0.235, 0.80)#(cmap, 0.235, 0.80)
    x = np.linspace(0, 1, 256, endpoint=False)
    terrain_cspace = np.fliplr(np.asarray([ground_cmap(i)[:3] for i in x]) * 255).astype(np.uint8)
    
    terrain_r_t = np.asarray([terrain_cspace[i][0] for i in terrain.flatten()]).reshape(terrain.shape)
    terrain_g_t = np.asarray([terrain_cspace[i][1] for i in terrain.flatten()]).reshape(terrain.shape)
    terrain_b_t = np.asarray([terrain_cspace[i][2] for i in terrain.flatten()]).reshape(terrain.shape)

    terrain_t = cv2.merge([terrain_r_t, terrain_g_t, terrain_b_t])

    # cv2.imshow("water_t", water_t)
    # cv2.imshow("tree_t", tree_t)
    # cv2.imshow("terrain_t", terrain_t)

    wt_t = np.maximum(water_t, terrain_t)
    # cv2.imshow('water_terrain', wt_t)
    all_t = np.maximum(wt_t, tree_t.astype(np.uint8))
    # print wt_t, tree_t
    # cv2.imshow('Total', all_t)
    # cv2.waitKey(10)
    return all_t

def get_roi(p, frame, half_size=16):
    if ((p[0] - half_size < 0 or p[0] + half_size >= frame.shape[1]) or \
        (p[1] - half_size < 0 or p[1] + half_size >= frame.shape[0])):
        # Pad image 
        # print '>>> Corner case!!!'
        frame = cv2.copyMakeBorder(frame,half_size,half_size,half_size,half_size,cv2.BORDER_CONSTANT,value=(255, 255, 255)) # wall colour
        p = (p[0]+half_size, p[1]+half_size)
    return frame[p[1]-half_size:p[1]+half_size, p[0]-half_size:p[0]+half_size]


if __name__ == '__main__':
    # m = draw_world([35.8689, -142.3132])
    # draw_world([0,0])
    # m = draw_terrain(shape=(120, 160))
    m = draw_terrain(shape=(250, 250))
    t = draw_trees(terrain = m, freq=30, R=3)
    cv2.imshow("map", m)
    cv2.imshow("trees", t)
    cv2.imshow('Pretty map', get_pretty_map(m, t))
    cv2.imwrite('pretty_map.png', get_pretty_map(m, t))
    cv2.waitKey(0)
