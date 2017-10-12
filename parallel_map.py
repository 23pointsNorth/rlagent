import multiprocessing
from functools import partial

import terrainEnv as te
import numpy as np

def merge_names(a, b):
    return '{} & {}'.format(a, b)

def env_resets(env):
    _, a = env.reset()
    return a

if __name__ == '__main__':
    pars = 20
    envs = [te.TerrainEnv() for i in xrange(pars)]
    pool = multiprocessing.Pool(processes=3)
    results = pool.map(env_resets, envs)
    pool.close()
    pool.join()    
    print(results)
    print (np.vstack(results))