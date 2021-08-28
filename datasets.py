# %%
import numpy as np
from scipy.stats import multivariate_normal 
from typing import List, Tuple, Union
import matplotlib.pyplot as plt

# %%

def gaussian_blob(n: int , mean : Union[List,np.ndarray] , cov : Union[List, np.ndarray]) -> np.ndarray:
    """Generates N-dimensional n gaussian samples using the scipy library.
    """
    norm = multivariate_normal(mean=mean, cov=cov)
    blob = norm.rvs(n)
    return blob










# %%
