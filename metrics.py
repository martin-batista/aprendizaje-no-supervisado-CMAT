# %%
import numpy as np
from sklearn.metrics import euclidean_distances

# %%

def l1_distance(x : np.ndarray, y : np.ndarray) -> np.ndarray:
    return np.sum(np.abs(x - y))

def l2_distance(x : np.ndarray, y : np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.power(x - y, 2)))

def l2_squared_distance(x : np.ndarray, y : np.ndarray) -> np.ndarray:
    return np.sum(np.power(x - y, 2))

def lp_distance(x : np.ndarray, y : np.ndarray, p : int = 2) -> np.ndarray:
    return np.power(np.sum(np.power(x - y, p)), 1/p)

def inf_distance(x : np.ndarray, y : np.ndarray) -> np.ndarray:
    return np.max(np.abs(x - y))

# %%
## Joaquim Viegas fast implementation of Dunn index.
## https://github.com/jqmviegas/jqm_cvi/tree/master/jqmcvi

def delta_fast(ck, cl, distances):
    values = distances[np.where(ck)][:, np.where(cl)]
    values = values[np.nonzero(values)]

    return np.min(values)
    
def big_delta_fast(ci, distances):
    values = distances[np.where(ci)][:, np.where(ci)]
    #values = values[np.nonzero(values)]
            
    return np.max(values)

def dunn(points, labels):
    """ Dunn index - FAST (using sklearn pairwise euclidean_distance function)
    
    Parameters
    ----------
    points : np.array
        np.array([N, p]) of all points
    labels: np.array
        np.array([N]) labels of all points
    """
    distances = euclidean_distances(points)
    ks = np.sort(np.unique(labels))
    
    deltas = np.ones([len(ks), len(ks)])*1000000
    big_deltas = np.zeros([len(ks), 1])
    
    l_range = list(range(0, len(ks)))
    
    for k in l_range:
        for l in (l_range[0:k]+l_range[k+1:]):
            deltas[k, l] = delta_fast((labels == ks[k]), (labels == ks[l]), distances)
        
        big_deltas[k] = big_delta_fast((labels == ks[k]), distances)

    di = np.min(deltas)/np.max(big_deltas)
    return di
# %%
