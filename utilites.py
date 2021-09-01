# %%
import re
from typing import Callable
from collections.abc import Iterable

import networkx as nx
from networkx.algorithms.components import connected
from networkx.algorithms.connectivity.kcomponents import k_components
from networkx.algorithms.distance_measures import eccentricity
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import combinations, cycle, islice
from typing import List
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm

import scipy
import math
from scipy.spatial import Delaunay
from sklearn import cluster
from visualizations import make_plot, make_cmap, plot_data
from scipy.spatial import Voronoi, voronoi_plot_2d
from metrics import l2_distance 

from typing import Union, List
from scipy.stats import multivariate_normal, bernoulli
from sklearn.cluster import KMeans
from metrics import dunn
# %matplotlib

# %%

class GraphTools():
    #    This class is a uses NetworkX, Sklearn and Scipy to construct and measure different properties of
    #    spatial neighborhood graphs for the purpose of assesing cluster validity.

    def __init__(self, points : np.ndarray = None, clusterer : Callable = None, metric : Callable = l2_distance) -> None:
        if points is not None:
            self.points = points
            self._delaunay_simplices = self._delaunay_triangulation(points).simplices
        
        self.clusterer = clusterer
        self._gabriel_graph = None
        self._delaunay_graph = None
        self.metric = metric
    
    # Graph node and edge insertion.
    def _graph_insert_nodes(self, G : nx.Graph, points : np.ndarray, clusterer : Callable = None) -> nx.Graph:
        for n, (x, y) in enumerate(points):
            if clusterer is not None:
                cluster_n = clusterer.predict([(x,y)])[0]
                G.add_node(n, pos=(x,y), cluster = cluster_n)
            else:
                G.add_node(n, pos=(x,y))
        return G
    
    def _graph_insert_edge(self, G : nx.Graph, edges : Iterable, points : np.ndarray, clusterer : Callable = None) -> nx.Graph:
        for edge in edges:
            weight = self.metric(points[edge[0]], points[edge[1]])
            if clusterer is not None:
                x_cluster_num = clusterer.predict([points[edge[0]]])[0]
                y_cluster_num = clusterer.predict([points[edge[1]]])[0]

                if x_cluster_num == y_cluster_num:
                    edge_type = 'internal'
                else:
                    edge_type = 'external'
                
                G.add_edge(*edge, weight=weight, edge_type=edge_type)
            else:
                G.add_edge(*edge, weight=weight)
        return G
    
    ## Graph constructors.
    def _delaunay_triangulation(self, points) -> scipy.spatial.qhull.Delaunay:
        """
           Calculates the Delaunay triangulation from the scipy.spatial.quhull.Delaunay implementation. 
           The Delaunay triangulation places edges every three points (x, y, z) such as no other points are
           contained in the circumscribed circle by the x,y,z simplex. 
        """
        assert points is not None, "No points!"
        return Delaunay(points)
    
    def delaunay_graph(self, points : np.ndarray = None, clusterer : Callable = None) -> nx.Graph:
        """
           Constructs the Delaunay graph. The graph places nodes at every point and edges every three points (x, y, z) such as no other points are
           contained in the circumscribed circle by the x,y,z simplex. The weight of the edge x-y is the distance metric of the points.
           The Delaunay triangulation is constructed through scipy.spatial.qhull.Delaunay where the L2 metric is used.

           Parameters:
           -----------
                points : [np.ndarray] An array of n-dimensional points.
                clusterer : [optional, Callable] A clustering class to assign 'cluster' label for the nodes.
                            clusterer.predict() method is called to obtain the abels.
           Returns:
           --------
                Graph: [nx.Graph] A networkX graph where the nodes are the points and the edges are weighted by
                                  the distance between them. 
        """
        if points is not None:
            self.__init__(points, clusterer)
        elif self._delaunay_graph is not None:
            return self._delaunay_graph
            
        G = nx.Graph(name='Delaunay triangulation')
        G = self._graph_insert_nodes(G, self.points, self.clusterer)

        # Insert edges.
        for path in self._delaunay_simplices: #Delaunay simplices: [[x_idx, y_idx, z_idx], ...]. 
            edges = combinations(path,2)
            G = self._graph_insert_edge(G, edges, self.points, self.clusterer)
        self._delaunay_graph = G
        return G

    def _inside_midpoint_circle(self, x : np.ndarray, y : np.ndarray , point : np.ndarray) -> bool:
        """
           Evalutes whether or not point is inside the circle of diameter xy.
           Utilizes the euclidean (L2) metric.
        """
        diam_sq = np.sum(np.power(x-y, 2))
        return diam_sq > np.sum(np.power(x-point,2)) + np.sum(np.power(y-point,2)) 

    @property 
    def _gabriel_edges(self) -> set:
        """
          Gabriel edges constructor. Depends on scipy.spatial.qhull.Delaunay. Iterates through the simplices
          and asseses whether or not the angles are acute or obtuse. If the angle ABC> is obtuse then AC is an 
          edge of the Gabriel graph.
          Complexity: O(n^2).
          To-do:
            Implement the graph in O(n*log(n)) time by checking if the edge of two nodes AB crosses the Voronoi boundary in common to 
            the nodes.
            E.g if the boundary is not common to the Voronoi regions of A and B, then remove the edge.
        """
        assert self.points is not None, "No points!"
        removed_edges = [] 
        edges = []
        simplices = self._delaunay_simplices
        points = self.points

        for simplex in simplices: #Iterate over simplices.
            for edge in combinations(simplex, 2): #Iterate over edges in the simplex.
                if edge not in removed_edges:
                    edges.append(frozenset(edge))
                    edge_point_x, edge_point_y = points[edge[0]], points[edge[1]]
                    for neighbor in set(simplex).difference(set(edge)):
                        neighbor_point = points[neighbor]
                        if self._inside_midpoint_circle(edge_point_x, edge_point_y, neighbor_point):
                            removed_edges.append(frozenset(edge))
                            
        edges = set(edges).difference(set(removed_edges))
        edges = [tuple(fset) for fset in edges]
        return edges 
    
    def gabriel_graph(self, points : np.ndarray = None, clusterer : Callable = None) -> set:
        """
           Constructs the Gabriel graph. The graph palces an edge every pair of points (x, y) if no other
           point is contained in the circle with diameter |x - y|. The edges are weighted by 
           the distance between them.
           The Gabriel graph is a subgraph of the Delaunay triangulation, which implicitly utilizes the L2 metric.

           Parameters:
           -----------
                points : [np.ndarray] An array of n-dimensional points.
                clusterer : [optional, Callable] A clustering class to assign 'cluster' label for the nodes.
                            clusterer.predict() method is called to obtain the abels.
           Returns:
           --------
                Graph: [nx.Graph] A networkX graph where the nodes are the points and the edges are weighted by
                                  the distance between them. 
        """
        if points is not None:
            self.__init__(points, clusterer)
        elif self._gabriel_graph is not None:
            return self._gabriel_graph
        
        G = nx.Graph(name = 'Gabriel graph')
        G = self._graph_insert_nodes(G, self.points, self.clusterer)
        G = self._graph_insert_edge(G, self._gabriel_edges, self.points, self.clusterer)
        self._gabriel_graph = G
        return G

    def get_cluster_subgraph(self, G : nx.Graph, n_cluster : int) -> nx.Graph:
        """
           Returns the subgraph of G that contains the nodes of the cluster n.
           
           Paramters:
           ----------
                G : [nx.Graph] A networkX graph that contains nodes with the attribute: "cluster".
                n_cluster : [int] The cluster number to filter.

           Returns:
           --------
                 nG : [nx.Graph] The subgraph that contains only the nodes belonging to the cluster n.
        """
        nodes_list = []
        for node, attributes_dict in G.nodes(data=True):
            if attributes_dict['cluster'] == n_cluster:
                nodes_list.append(node)
        nG = G.subgraph(nodes_list) 
        return nG 
    
    def get_pair_transition_subgraph(self, G : nx.Graph, n_cluster_1 : int, n_cluster_2 : int) -> nx.Graph:
        """
           Returns the subgraph of G that contains the nodes that transition from the cluster n_cluster_1
           to the cluster n_cluster_2.
           
           Paramters:
           ----------
                G : [nx.Graph] A networkX graph that contains nodes with the attribute: "cluster".
                n_cluster_1, n_cluster_2 : [int] The cluster numbers. 

           Returns:
           --------
                 nG : [nx.Graph] The subgraph that contains only that begin in n_cluster_1 and end in n_cluster_2.
        """
        nG = nx.DiGraph()
        nodes_data = G.nodes(data=True)
        for node, attributes_dict in nodes_data: 
            if attributes_dict['cluster'] == n_cluster_1:
               for edge in G.edges(node): 
                   edge_data = G.get_edge_data(*edge)
                   if edge_data['edge_type'] == 'external' and nodes_data[edge[1]]['cluster'] == n_cluster_2:
                    nG.add_node(edge[0], **nodes_data[edge[0]])
                    nG.add_node(edge[1], **nodes_data[edge[1]])
                    nG.add_edge(edge[0], edge[1], **edge_data)

        return nG 
    
    def get_transitions_subgraph(self, G : nx.Graph, n_cluster : int) -> nx.Graph:
        """
           Returns the subgraph of G that contains the nodes that transition from the cluster n_cluster
           to other clusters. 
           
           Paramters:
           ----------
                G : [nx.Graph] A networkX graph that contains nodes with the attribute: "cluster".
                n_cluster : [int] The cluster number. 

           Returns:
           --------
                 nG : [nx.Graph] The subgraph that contains only that begin in n_cluster_1 and end in n_cluster_2.
        """
        nG = nx.DiGraph()
        nodes_data = G.nodes(data=True)
        for node, attributes_dict in nodes_data: 
            if attributes_dict['cluster'] == n_cluster:
               for edge in G.edges(node): 
                   edge_data = G.get_edge_data(*edge)
                   if edge_data['edge_type'] == 'external':
                    nG.add_node(edge[0], **nodes_data[edge[0]])
                    nG.add_node(edge[1], **nodes_data[edge[1]])
                    nG.add_edge(edge[0], edge[1], **edge_data)

        return nG 
    
    def get_diameter_path(self, G : nx.Graph) -> nx.Graph:
        """
           Builds the diameter subgraph of a given graph. The diameter is defined as the longest geodesic
           inside the graph.

           Paramteres:
           -----------
                G : [nx.Graph] The graph to calculate the diamter from.

           Returns: 
           --------
                dG : [nx.Graph] The subgraph that corresponds to the longest geodesic inside 
                                the graph.
        """
        if not G.is_directed():
            pairwise_shortest_distances = nx.all_pairs_dijkstra_path_length(G)
            eccentricity = nx.eccentricity(G, sp=dict(pairwise_shortest_distances))
            sorted_shortest_paths = [(k,v) for k, v in sorted(eccentricity.items(), key=lambda x: x[1], reverse=True)]
            source = sorted_shortest_paths[0][0]
            target = sorted_shortest_paths[1][0]
            nodes = nx.shortest_path(G, source, target)
        else:
            nodes = nx.dag_longest_path(G)

        return nodes
    
    def get_shortest_path(self, G : nx.Graph) -> nx.Graph:
        """
           Returns the shortest path in the graph. 

           Paramters:
           ----------
                G : [nx.Graph] A netwrokX graph.
            
           Returns:
           --------
                shortest_distance : [float] The shortest distance inside the graph.
           
           Raises:
           -------
                Assertion Error if the graph is empty.
        """
        assert not nx.is_empty(G), "Empty graph!"
        pairwise_shortest_distances = nx.all_pairs_dijkstra_path_length(G)
        shortest_distance = -1
        for node, path_dict in pairwise_shortest_distances:
            for path_node, distance in path_dict.items():
                if path_node != node:
                    if distance < shortest_distance or shortest_distance < 0:
                        shortest_distance = distance
                        shortest_target = path_node
                        shortest_source = node

        nodes = nx.dijkstra_path(G, shortest_source, shortest_target)
        return nodes

    def shortest_distance(self, G : nx.Graph) -> float:
        """
           Returns the shortest distance in the graph.
           If the graph is empty, it returns inf.

           Paramters:
           ----------
                G : [nx.Graph] A netwrokX graph.
            
           Returns:
           --------
                shortest_distance : [float] The shortest distance inside the graph.
        """
        if nx.is_empty(G):
            return math.inf
        else:
            pairwise_shortest_distances = nx.all_pairs_dijkstra_path_length(G)
            shortest_distance = -1
            for node, path_dict in pairwise_shortest_distances:
                for path_node, distance in path_dict.items():
                    if path_node != node:
                        if distance < shortest_distance or shortest_distance < 0:
                            shortest_distance = distance

        return shortest_distance

    def _connected_diameter(self, G : nx.Graph) -> nx.Graph:
        """
           Returns diameter of the weakly connected graph. 
           The diameter is the length of the largest geodesic inside the graph.

           Paramters:
           ----------
                G : [nx.Graph] The graph to calculate the diameter of.
           
           Returns:
           --------
                diameter : [float] The diameter of the graph.
        """
        d = nx.all_pairs_dijkstra_path_length(G)
        e = nx.eccentricity(G, sp=dict(d))
        diameter = nx.diameter(G, e)
        return diameter
    
    def diameter(self, G : nx.Graph) -> float:
        """
           The weighted diameter of the graph.
           For disconnected graphs, it returns the maximum diameter of the connected subgraphs.
           
           Paramters:
           ----------
                G : [nx.Graph] The graph to calculate the diameter of.
           
           Returns:
           --------
                diameter : [float] The diameter of the graph.
        """
        if nx.is_empty(G):
            return 0.
        else:
            if not G.is_directed():
                if nx.is_connected(G):
                    return self._connected_diameter(G)
                else:
                    diameter = 0
                    for connected_nodes in nx.connected_components(G):
                        connected_sG = G.subgraph(connected_nodes)
                        sG_diameter = self._connected_diameter(connected_sG) 
                        if sG_diameter > diameter:
                            diameter = sG_diameter
                    return diameter
            else:
                return nx.dag_longest_path_length(G)
    
    def cluster_diameter(self, G : nx.Graph, n_cluster : int) -> float: 
        """
           Returns the diameter of a specified cluster in graph G.

           Parameters:
           -----------
                G : [nx.Graph] The data graph.
                n_cluster : [int] The cluster number.
            
           Reutrns:
           --------
                min_cluster_distance : [float] The diameter (length of the largest geodesic) of a cluster in a given graph.
        """
        cG = self.get_cluster_subgraph(G, n_cluster)
        return self.diameter(cG)

    def min_cluster_distance(self, G : nx.Graph, n_cluster : int) -> float:
        """
            Returns the minimum graph distance from cluster n_cluster to another cluster.

           Parameters:
           -----------
                G : [nx.Graph] The data graph.
                n_clusters : [int] The number of clusters present in the graph.
            
           Reutrns:
           --------
                min_cluster_distance : [float] The minimum distance to an adjacent graph. 
        """
        tG = self.get_transitions_subgraph(G, n_cluster)
        return self.shortest_distance(tG)
    
    def modified_dunn_index(self, G : nx.Graph, n_clusters : int) -> float:
        """
           Calculates the modified Dunn index based on neighborhood graphs. The Dunn index is defined as the
           min_d(c_i,c_j)/max_D(c_k) for all i,j,k, where min_d is the cluster distance as measured by min_cluster_distance(G , n_cluster)
           and max_D(c_k) is the largest diameter of the clusters.
           The distance is calculated by traversals in the weighted graph.

           Parameters:
           -----------
                G : [nx.Graph] The data graph.
                n_clusters : [int] The number of clusters present in the graph.
            
           Reutrns:
           --------
                modified_dunn_index : [float] The modified version of the Dunn index, where distances are calculated in the graph.
        """
        max_diameter = 0
        min_distance = -1
        for n_cluster in range(n_clusters):
            _cluster_distance = self.min_cluster_distance(G, n_cluster)
            _cluster_diameter = self.cluster_diameter(G, n_cluster)
            if _cluster_distance < min_distance or min_distance < 0:
                min_distance = _cluster_distance
            if _cluster_diameter > max_diameter:
                max_diameter = _cluster_diameter
        
        return min_distance/max_diameter

    ## Graph plotting.

    def draw(self, G, fig = None, plot_size = 12, diameter = False, shortest_distance = False, diameter_color = 'firebrick' ,
             diameter_factor = 1.5, voronoi=False, voronoi_width = 1, voronoi_alpha = 0.6, 
             edge_weights = False, edge_width = 2.4, node_size = 25, node_color = None, edge_color = None, 
             voronoi_color = 'darkgrey', theme='dark'):
        """
           Draws the specified graph.
        """

        edge_width = edge_width
        pos=nx.get_node_attributes(G,'pos')
        if fig is None:
            plt.close()
            if theme == 'light':
                fig = plt.figure(figsize=(plot_size,plot_size), edgecolor='white', facecolor='white')
                plt.style.use('seaborn')
                plt.rc('axes',edgecolor='white')
            elif theme == 'dark':
                fig = plt.figure(figsize=(plot_size,plot_size), edgecolor='black', facecolor='black')
                plt.style.use('dark_background')
                plt.rc('axes',edgecolor='black')

            lst = [i for i, _ in enumerate(self.clusterer.cluster_centers_)]
            color_minima = lst[0]
            color_maxima = lst[-1]
            self._cmap = make_cmap(theme)
            self._color_cycler = cycle(lst) # Constructs a cycle iterator to pass on as a color for the regions.
            norm = matplotlib.colors.Normalize(vmin=color_minima, vmax=color_maxima, clip=True)
            self._color_mapper = cm.ScalarMappable(norm=norm, cmap=self._cmap)

        ax = fig.gca()

        if edge_color is None and node_color is None: 
            rgba_value = next(self._color_cycler)
            edge_color = self._color_mapper.to_rgba(rgba_value)
            r,g,b,a = edge_color
            if theme == 'dark':
                brightness = 1.7
                offset = 0.1
                r = min(1, r*brightness + offset)
                g = min(1, g*brightness + offset)
                b = min(1, b*brightness + offset)
            elif theme == 'light':
                brightness = 0.6
                offset = 0
                r = min(1, r*brightness + offset)
                g = min(1, g*brightness + offset)
                b = min(1, b*brightness + offset)
            node_color = [(r,g,b,a)]*len(G.nodes())
        elif edge_color is None:
            edge_color = node_color
        elif node_color is None:
            node_color = edge_color

        # Draw Voronoi cells:
        if voronoi:
            vor = Voronoi(self.points)
            voronoi_plot_2d(vor, plt.gca(), show_vertices=False, line_colors=voronoi_color,
                            line_alpha = voronoi_alpha, line_width=voronoi_width, point_size=0)

        # Draw the graph:
        nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=node_size, ax=ax) 
        if edge_weights is True:
            weights = list(nx.get_edge_attributes(G,'weight').values())
            rev_weights = list(np.abs(np.array(weights) - (max(weights) + min(weights)))*edge_width)
            nx.draw_networkx_edges(G, pos, edge_color=edge_color, ax=ax, width=rev_weights) 
        else:
            nx.draw_networkx_edges(G, pos, edge_color=edge_color, ax=ax, width=edge_width) 
        
        # Draw diameter:
        if diameter:
            path = self.get_diameter_path(G)
            path_edges = list(zip(path,path[1:]))
            nx.draw_networkx_nodes(G,pos,nodelist=path, node_color=diameter_color, node_size = node_size*diameter_factor)
            nx.draw_networkx_edges(G,pos,edgelist=path_edges, edge_color=diameter_color, width=edge_width*diameter_factor)

        if shortest_distance:
            path = self.get_shortest_path(G)
            path_edges = list(zip(path,path[1:]))
            nx.draw_networkx_nodes(G,pos,nodelist=path, node_color=diameter_color, node_size = node_size*diameter_factor)
            nx.draw_networkx_edges(G,pos,edgelist=path_edges, edge_color=diameter_color, width=edge_width*diameter_factor)

        plt.grid(True,c='darkgrey', alpha=0.3)
        plt.axis('on',) 
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.xlabel("$x_1$", fontsize=plot_size + 3)
        plt.ylabel("$x_2$", fontsize=plot_size + 3)
        plt.xticks(fontsize=plot_size)
        plt.yticks(fontsize=plot_size)
        plt.title(G.name, fontsize=plot_size + plot_size/2)
    
        return fig, plt

# %%
