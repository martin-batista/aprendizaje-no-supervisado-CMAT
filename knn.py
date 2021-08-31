# %%
# Librerias utilizadas:
from metrics import l2_distance
import numpy as np
from typing import Union, List
from scipy.stats import multivariate_normal, bernoulli

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from itertools import combinations

from spatial import GraphTools
from visualizations import plot_clusters, make_figure, make_plot

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
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(data)

    plot_clusters(data, kmeans)

# %%
gt = GraphTools(data)
gt.plot_all()
gt.plot_delaunay()
gt.plot_gabriel()
# gt.plot_all()


# %%
import networkx as nx

# %%
# Delaunay graph.
gt = GraphTools(data)

G = nx.Graph()
weighted_edge_list = []
for path in gt.simplices: #Delaunay simplices.
    edges = combinations(path,2)
    for edge in edges:
        weight = l2_distance(data[edge[0]], data[edge[1]])
        G.add_edge(*edge, weight=weight)

for n, (x, y) in enumerate(data):
    G.add_node(n, pos=(x,y))

pos=nx.get_node_attributes(G,'pos')

# %%

weights = list(nx.get_edge_attributes(G,'weight').values())
edge_thickness = 0.6
rev_weights = list(np.abs(np.array(weights) - (max(weights) + min(weights)))*edge_thickness)
fig = plt.figure(figsize=(12,12))
ax = fig.gca()

nx.draw_networkx_nodes(G, pos, node_color='lavenderblush', node_size=15, ax=ax) #notice we call draw, and not draw_networkx_nodes
nx.draw_networkx_edges(G, pos, edge_color='hotpink', ax=ax, width=rev_weights) #notice we call draw, and not draw_networkx_nodes
plt.grid(True,c='darkgrey', alpha=0.3)
plt.axis('on') # turns on axis
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
plt.xlabel("$x_1$", fontsize=14)
plt.ylabel("$x_2$", fontsize=14)
plt.title('Delaunay triangulation', fontsize=16)
plt.show()

# %%
# Gabriel graph.
G = nx.Graph()
weighted_edge_list = []
for edge in gt.gabriel_edges: #Delaunay simplices.
    weight = l2_distance(data[edge[0]], data[edge[1]])
    G.add_edge(*edge, weight=weight)

for n, (x, y) in enumerate(data):
    G.add_node(n, pos=(x,y))

pos=nx.get_node_attributes(G,'pos')

# %%
from scipy.spatial import Voronoi, voronoi_plot_2d

# %%
weights = list(nx.get_edge_attributes(G,'weight').values())
edge_thickness = 1.5
rev_weights = list(np.abs(np.array(weights) - (max(weights) + min(weights)))*edge_thickness)
fig = plt.figure(figsize=(24,24))
ax = fig.gca()
vor = Voronoi(data)
voronoi_plot_2d(vor, plt.gca(), show_vertices=False, line_colors="darkgrey",
                line_alpha = 0.6, line_width=1, point_size=0)

nx.draw_networkx_nodes(G, pos, node_color='cyan', node_size=15, ax=ax) #notice we call draw, and not draw_networkx_nodes
nx.draw_networkx_edges(G, pos, edge_color='palegreen', ax=ax, width=rev_weights) #notice we call draw, and not draw_networkx_nodes
plt.grid(True,c='darkgrey', alpha=0.3)
plt.axis('on') # turns on axis
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
plt.xlabel("$x_1$", fontsize=28)
plt.ylabel("$x_2$", fontsize=28)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Gabriel graph', fontsize=36)
plt.show()

# %%
# sp = nx.single_source_dijkstra_path_length
d = nx.all_pairs_dijkstra_path_length(G)
e = nx.eccentricity(G, sp=dict(d))
diameter = nx.diameter(G, e)
print('Diameter', diameter)

# %%
sorted_shortest_paths = [(k,v) for k, v in sorted(e.items(), key=lambda x: x[1], reverse=True)]
shortest_paths = list(nx.all_pairs_dijkstra_path(G))
longest_geodesic = shortest_paths[sorted_shortest_paths[0][0]]

# %%
weights = list(nx.get_edge_attributes(G,'weight').values())
edge_thickness = 1.5
rev_weights = list(np.abs(np.array(weights) - (max(weights) + min(weights)))*edge_thickness)
fig = plt.figure(figsize=(24,24))
ax = fig.gca()
vor = Voronoi(data)
voronoi_plot_2d(vor, plt.gca(), show_vertices=False, line_colors="darkgrey",
                line_alpha = 0.6, line_width=1, point_size=0)

nx.draw_networkx_nodes(G, pos, node_color='cyan', node_size=15, ax=ax) #notice we call draw, and not draw_networkx_nodes
nx.draw_networkx_edges(G, pos, edge_color='palegreen', ax=ax, width=rev_weights) #notice we call draw, and not draw_networkx_nodes

path = nx.shortest_path(G,source=180,target=123)
path_edges = list(zip(path,path[1:]))
nx.draw_networkx_nodes(G,pos,nodelist=path,node_color='tomato', alpha=0.6, node_size=40)
nx.draw_networkx_edges(G,pos,edgelist=path_edges,edge_color='tomato',alpha=0.6, width=4)

plt.grid(True,c='darkgrey', alpha=0.3)
plt.axis('on') # turns on axis
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
plt.xlabel("$x_1$", fontsize=28)
plt.ylabel("$x_2$", fontsize=28)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Gabriel graph', fontsize=36)


plt.show()
# %%
