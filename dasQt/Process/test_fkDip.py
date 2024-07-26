import numpy as np
import matplotlib.pyplot as plt

from fkDip import fkDip

def test_fkDip():
    d = np.random.rand(100, 100)
    w = 0.1
    d0 = fkDip(d, w)
    
    return d0

if __name__ == '__main__':
    d0 = test_fkDip()
    plt.imshow(d0, cmap='gray')
    plt.show()
