# %%
import numpy as np
from typing import List
import matplotlib.pyplot as plt

# %%

class Datasets():

    def __init__(self) -> None:
       pass 

    def make_gaussian_blobs(self, n: int, mu : np.ndarray , sigma : np.ndarray) -> np.ndarray:
        """Generates a gaussian random data. In the case of two dimensions it utilizes the Box-Mueller algorithm,
           otherwise it utilies the Cholesky algorithm.
           Params:
                - n : The ammount of points to be created.
                - mu : The mean (size N) vector. Type must be list or ndarray.
                - sigma : The NxN standard deviation matrix. Type must be list or ndarray.
        """
        mu = np.array(mu)
        sigma = np.array(sigma)

        #Enforce correct input: 
        assert isinstance(mu, np.ndarray) and isinstance(sigma, np.ndarray), "Couldn't convert input into ndarray."
        assert mu.shape[0] == sigma.shape[0],  "Invalid dimensions for mu and sigma."
        if sigma.shape[0] > 1:
            assert sigma.shape[0] == sigma.shape[1],  "Sigma matrix is not square."
        assert all(sigma.flatten() >= 0), "Sigma matrix is not positive semi-definite."

        ##Dimension 2, using Box-Mueller:
        u = np.random.random(n)
        v = np.random.random(n)
        bm_x = np.sqrt(-2*np.log(u))*np.sin(2*np.pi*v)
        bm_y = np.sqrt(-2*np.log(u))*np.cos(2*np.pi*v)








# %%
