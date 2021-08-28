# %%
# Librerias utilizadas:
from os import remove
import numpy as np
from typing import Callable, Union, List
from scipy.stats import multivariate_normal, bernoulli
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from spatial import GraphTools

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

# Creación de gráficas:

def make_plot() -> None:
    """Plot templates."""
    plt.figure(figsize=(12,12))
    plt.style.use('dark_background')
    plt.rc('axes',edgecolor='k')
    ax = plt.axes()
    ax.tick_params(axis='x', colors='darkgrey')
    ax.tick_params(axis='y', colors='darkgrey')

def make_cmap(colors : List = None) -> None:
    if colors is None:
        colors = ["hotpink", "orchid", "palegreen", "mediumspringgreen", "aqua", "dodgerblue"]
    cmap = LinearSegmentedColormap.from_list("custom_bright", colors)
    return cmap
 
def plot_data(data : np.ndarray, y_labels : np.ndarray = None, cmap : str = 'Set3', size : int = 15, alpha : float = 1) -> None:
    if y_labels is None:
        color = 'cyan'
    else:
        color = y_labels

    plt.scatter(data[:,0], data[:,1], c = color, marker='o', cmap=cmap, s=size, zorder= 14, alpha=alpha)
    plt.scatter(data[:,0], data[:,1], color='k', marker='o', cmap=cmap, s=size, zorder=1, alpha=0.6, linewidths=6)
    plt.grid(True,c='darkgrey', alpha=0.3)

def plot_centroids(centroids : np.ndarray, center_labels : np.ndarray, cmap: str, weights : float =None) -> None:
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=85, linewidths=12,
                # color='grey', cmap = 'Set3', zorder=10, alpha=0.5)
                c=center_labels, cmap = cmap, zorder=20, alpha=0.4)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=5, linewidths=17,
                color='k', cmap = cmap, zorder=25, alpha=1)
                # c=center_labels, cmap = 'Set3', zorder=11, alpha=1)

def plot_decision_boundaries(clusterer : Callable, X : np.ndarray, cmap:str, resolution : int =1000, show_centroids : bool =True,
                             show_xlabels : bool =True, show_ylabels :bool =True) -> None:
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    cluster_labels = clusterer.predict(X)
    center_labels = clusterer.predict(clusterer.cluster_centers_)
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap=cmap, alpha=0.2)
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X, cluster_labels, cmap)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_, center_labels, cmap)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)


# %%
if __name__ == '__main__':
    data = make_data()
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(data)

    make_plot() # Grafica nueva.
    cmap = make_cmap()
    plot_decision_boundaries(kmeans, data, cmap)

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
