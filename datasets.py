# %%
import numpy as np
from scipy.stats import multivariate_normal 
from typing import List
import matplotlib.pyplot as plt

# %%


def gaussian_2D_blob(n: int , mean : np.ndarray , cov : np.ndarray) -> np.ndarray:
    norm = multivariate_normal(mean=mean, cov=cov)
    blob = norm.rvs(n)
    return blob[:,0], blob[:,1]








# %%
