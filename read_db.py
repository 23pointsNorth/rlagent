import h5py    # HDF5 support
import numpy as np

f = h5py.File('data.h5', "r")

print f['img'].shape
print f['angle'].shape
print f['goal_angle'].shape
