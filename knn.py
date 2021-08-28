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

    plot_clusters(data, kmeans, True)

# %%
gt = GraphTools(data)
gt.plot_all()
# gt.plot_delaunay()
# gt.plot_gabriel()
# gt.plot_all()

# %%
from itertools import combinations
from scipy.spatial import Delaunay
from time import perf_counter
from numba import jit
import math

points = make_data(n=300)
# points = np.array([[1,2], [3,5], [7,3], [1,5], [4,2]])
make_plot()
plot_data(points)
t1 = perf_counter()
tri = Delaunay(points)
t2 = perf_counter()
print('Delaunay', t2-t1)
plt.triplot(points[:,0], points[:,1], tri.simplices, color='hotpink')


def inside_circle(x : np.ndarray, y : np.ndarray , point : np.ndarray) -> bool:
    """Evalutes whether or not point is inside the circle of diameter xy."""
    # diam = l2_squared_distance(x,y)
    # return diam**2 > l2_squared_distance(x,point) ** 2 + l2_squared_distance(y,point) **2
    diam_sq = np.sum(np.power(x-y, 2))
    return diam_sq > np.sum(np.power(x-point,2)) + np.sum(np.power(y-point,2)) 

def gabriel(simplices):
    removed_edges = [] 
    edges = []
    simplices = tri.simplices
    for simplex in simplices: #Iterate over simplices.
        for edge in combinations(simplex, 2): #Iterate over edges in the simplex.
            if edge not in removed_edges:
                edges.append(frozenset(edge))
                edge_point_x, edge_point_y = points[edge[0]], points[edge[1]]
                for neighbor in set(simplex).difference(set(edge)):
                    neighbor_point = points[neighbor]
                    if inside_circle(edge_point_x, edge_point_y, neighbor_point):
                        removed_edges.append(frozenset(edge))
                        
    # removed_edges = [tuple(removed_edge) for removed_edge in removed_edges]
    edges = set(edges).difference(set(removed_edges))
    # edges = np.array(list(edges))
    return edges

t1 = perf_counter()
edges = gabriel(tri.simplices)
t2 = perf_counter()

print('Gabriel', t2-t1)
# edges_ = np.array([list(edge) for edge in edges])
# make_plot()
# plot_edges = np.array([list(list(edge)) for edge in edges])
for edge in edges:
    plt.plot(points[[*edge]][:,0], points[[*edge]][:,1], '-', color='palegreen')
    # plt.plot(points[edge][:,0], points[edge][:,1], '-', color='palegreen')
    # plt.plot(points[[*edge]][:,0], points[[*edge]][:,1], '-', color='hotpink')
    plot_data(points[[*edge]], size=40)
    # plot_data(points[edge], size=40)

# plt.show()




# def gabriel_graph(points : List) -> List:


# %%
