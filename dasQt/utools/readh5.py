import numpy as np
import h5py

def readh5(filename):
    with h5py.File(filename, 'r') as f:
        for a in f:
            StrainRate = f[f'/{a}/Source1/Zone1/StrainRate'][:]
            spacing = f[f'/{a}/Source1/Zone1'].attrs['Spacing']


    #self.data = StrainRate.reshape(-1, StrainRate.shape[-1])
    dx          = spacing[0]
    dt          = spacing[1] * 1e-3
    nb, nt, nx  = StrainRate.shape
    data        = StrainRate[:, nt//2:, :].reshape(-1, nx)
    nt, nx      = data.shape
    gauge       = dx    #TODO: check this
    
    metadata = {'fs': 1/dt, 
                'dt': dt,
                'dx': dx,
                'nx': nx,
                'nt': nt,
                'gauge': gauge}

    return data, metadata

