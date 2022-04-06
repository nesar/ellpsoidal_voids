import numpy as np
## only creating ellpsoids for training now. Will extend it to arbitrary geometries later
from create_ellipsoids import *


np.random.seed(1)

rmin = 0.0
rmax = 0.004
ngrid = 128

nsamples = 1000

rx_all = np.random.uniform(low=rmin, high=rmax, size=(nsamples,))
ry_all = np.random.uniform(low=rmin, high=rmax, size=(nsamples,))
rz_all = np.random.uniform(low=rmin, high=rmax, size=(nsamples,))

all_ellipsoids = np.zeros(shape=(nsamples, ngrid, ngrid, ngrid))
for indx in range(rx_all.shape[0]):
    ellipsoid = create_geometry(rx_all[indx], ry_all[indx], rz_all[indx], rmin, rmax, ngrid)
    all_ellipsoids[indx, :, :, :] = ellipsoid

outfile = '../data/random_ellipsoids_3d'
np.save(outfile, all_ellipsoids)
np.save(outfile + '_radii', np.array([rx_all, ry_all, rz_all]).T )



