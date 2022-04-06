import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
plt.rcParams.update({'font.size': 22})

##################################################

fileIn = '../data/DOE1.out'

allData = np.loadtxt(fileIn, skiprows=1)
allLabels = ['rx', 'ry', 'rz', 'dist', 'Smax', 'Vol of high stress']
#allData[:, -1] = np.log10(allData[:, -1])

all_df = pd.DataFrame(allData)
all_df.columns = allLabels


'''
FROM PIYUSH

The four parameters are the three semi-axis of the ellipsoid void, Rx, Ry, Rz 
and the distance between the center of the void and edge of the space named ‘dist’.

The outputs to be modeled are maximum stress  (Smax, in column 5), 
and the volume on which the stress is between 75%of max stress and max stress (Vol of high stress in column 6).
'''

###########################################

plt.figure(figsize=(25,25))
g = sns.pairplot(all_df, kind='scatter', plot_kws=dict(alpha=0.3, size=0.25) )
g.map_lower(sns.kdeplot, levels=3, color=".3")
plt.savefig('../plots/read_allData.png', bbox_inches='tight')

#plt.tight_layout()
#plt.show()
###########################################



from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
ax = fig.add_subplot(111, projection='3d')

#coefs = (1, 2, 2)  # Coefficients in a0/c x**2 + a1/c y**2 + a2/c z**2 = 1
# Radii corresponding to the coefficients:
#rx, ry, rz = 1/np.sqrt(coefs)

ellIdx = 12
rx, ry, rz = allData[ellIdx, :3]

# Set of all spherical angles:
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

# Cartesian coordinates that correspond to the spherical angles:
# (this is the equation of an ellipsoid):
x = rx * np.outer(np.cos(u), np.sin(v))
y = ry * np.outer(np.sin(u), np.sin(v))
z = rz * np.outer(np.ones_like(u), np.cos(v))

# Plot:
ax.plot_surface(x, y, z,  rstride=4, cstride=4, cmap=cm.gray, linewidth=1, antialiased=False)

# Adjustment of the axes, so that they all have the same span:
max_radius = max(rx, ry, rz)
for axis in 'xyz':
    getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))

plt.show()


###############################################


rmin = 0.0005
rmax = 0.0025

rmin = 0.0
rmax = 0.004

ngrid = 100

xgrid,ygrid,zgrid = np.meshgrid(np.linspace(rmin, rmax, ngrid), np.linspace(rmin, rmax, ngrid), np.linspace(rmin, rmax, ngrid))
#xgrid,ygrid,zgrid = np.meshgrid(np.linspace(-1, 1, ngrid), np.linspace(-1, 1, ngrid), np.linspace(-1, 1, ngrid))
xgrid_cen = ygrid_cen = zgrid_cen = (rmax - rmin)/2.
# filter points outside ellipsoid interior:
#R = (rmax - rmin)/2

#mask = (2*xgrid)**2 + (3*ygrid)**2 + zgrid**2 <= R**2

ell_eqn = ((xgrid - xgrid_cen)**2/(rx)**2) +  ((ygrid - ygrid_cen)**2/(ry)**2) + ((zgrid - zgrid_cen)**2/(rz)**2)
mask = ell_eqn <= 1
xgrid = xgrid[mask]
ygrid = ygrid[mask]
zgrid = zgrid[mask]


# convert to cartesian for plotting:
'''
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter3D(xgrid,ygrid,zgrid)
#ax.set_xlim(-1.2,1.2)
#ax.set_ylim(-1.2,1.2)
#ax.set_zlim(-1.2,1.2)
plt.show()
'''


mask01 = mask*1

f, a = plt.subplots(1,3, figsize = (12, 4))
a[0].imshow(mask01[50, :, :])
a[1].imshow(mask01[:, 50, :])
a[2].imshow(mask01[:, :, 50])

plt.show()


##########################################

from create_ellipsoids import *

create_geometry(allData[10, 0], allData[10, 1], allData[10, 2], rmin = 0.0, rmax = 0.004, ngrid = 32)
