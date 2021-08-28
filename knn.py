# %%
# Librerias utilizadas:
import numpy as np
from typing import Union, List
from scipy.stats import multivariate_normal, bernoulli

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from spatial import GraphTools
from visualizations import plot_clusters

# %%
# Creación de datos. 
def gaussian_blob(n: int , mean : Union[List,np.ndarray], cov : Union[List, np.ndarray] = None) -> np.ndarray:
    """Generates N-dimensional n gaussian samples using the scipy library. 
       If cov is None it defaults to the identity."""

    norm = multivariate_normal(mean=mean, cov=cov)
    blob = norm.rvs(n)
    return blob

def make_data(n : int = 300) -> np.ndarray:
    """Samples from 0.5*N(0,I) and 0.5*N((4,4), I) given a bernoulli trial."""

    bernoulli_trials = np.sum(bernoulli.rvs(0.5, size=n))
    normal_1_trials = bernoulli_trials 
    normal_2_trials = n - bernoulli_trials 

    normal_1_points = 0.5*gaussian_blob(normal_1_trials, mean=[0,0],)
    normal_2_points = 0.5*gaussian_blob(normal_2_trials, mean=[4,4],) 

    return np.concatenate([normal_1_points, normal_2_points], axis=0)

# %%
if __name__ == '__main__':
    data = make_data()
    kmeans = KMeans(n_clusters=28)
    kmeans.fit(data)

    plot_clusters(data, kmeans)

# %%
gt = GraphTools(data)
gt.plot_all()
gt.plot_delaunay()
gt.plot_gabriel()
# gt.plot_all()


# %%
