import numpy as np

def create_geometry(rx, ry, rz, rmin, rmax, ngrid):

    xgrid,ygrid,zgrid = np.meshgrid(np.linspace(rmin, rmax, ngrid), np.linspace(rmin, rmax, ngrid), np.linspace(rmin, rmax, ngrid))
    xgrid_cen = ygrid_cen = zgrid_cen = (rmax - rmin)/2.

    ell_eqn = ((xgrid - xgrid_cen)**2/(rx)**2) +  ((ygrid - ygrid_cen)**2/(ry)**2) + ((zgrid - zgrid_cen)**2/(rz)**2)
    mask = ell_eqn <= 1

    '''
    xgrid = xgrid[mask]
    ygrid = ygrid[mask]
    zgrid = zgrid[mask]
    '''
    mask01 = mask*1
    return mask01


def create_arbitrary_geometry(rmin, rmax, ngrid):
    raise NotImplemented