# %%
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from typing import List
from matplotlib.colors import LinearSegmentedColormap
import scipy
from scipy.spatial import Delaunay
from visualizations import make_plot, make_cmap, plot_data

# %%

class GraphTools():

    def __init__(self, points : np.ndarray = None, metric : Callable = None ) -> None:
        if points is not None:
            self.points = points
            self.triangulation = self._delaunay_triangulation(points)
            self.simplices = self.triangulation.simplices
            self.gabriel_edges = self._gabriel_graph(points)
        self.metric = metric
    
    ## Graph constructors.
    def _delaunay_triangulation(self, points) -> scipy.spatial.qhull.Delaunay:
        """Constructs the Delaunay triangulation.
        """
        assert points is not None, "No points!"
        return Delaunay(points)

    def delaunay(self, points : np.ndarray = None) -> scipy.spatial.qhull.Delaunay:
        """Calculates the Delaunay triangulation from the scipy.spatial.quhull.Delaunay implementation. 
           The Delaunay triangulation places edges every three points (x, y, z) such as no other points are
           contained in the circumscribed circle by the x,y,z simplex. 
        """
        if points is not None:
            self.__init__(points)
        return self.triangulation
        
    def _inside_midpoint_circle(self, x : np.ndarray, y : np.ndarray , point : np.ndarray) -> bool:
        """Evalutes whether or not point is inside the circle of diameter xy.
           Utilizes the euclidean (L2) metric.
        """
        diam_sq = np.sum(np.power(x-y, 2))
        return diam_sq > np.sum(np.power(x-point,2)) + np.sum(np.power(y-point,2)) 
    
    def _gabriel_graph(self, points) -> set:
        """Gabriel graph constructor. Depends on scipy.spatial.qhull.Delaunay. 
        """
        assert points is not None, "No points!"
        removed_edges = [] 
        edges = []
        tri = self.triangulation 
        simplices = tri.simplices

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
        return edges
    
    def gabriel_graph(self, points = None) -> set:
        """The Gabriel graph palces an edge every pair of points (x, y) if no other
           point is contained in the circle with diameter |x - y|.
           It utilizes the euclidean (L2) metric to calculate distances.
           Returns: A set where each entry is a pair of indexes {a,b} corresponding to connected vertices.
        """
        if points is not None:
            self.__init__(points)
        return self.gabriel_edges
    
    ## Graph plotting.

    def plot_gabriel(self, points : np.ndarray = None) -> None:
        if points is not None:
            self.__init__(points)
 
        plot_data(self.points)
        for edge in self.gabriel_edges:
            plt.plot(self.points[[*edge]][:,0], self.points[[*edge]][:,1], '-', color='palegreen')
            # self._plot_data(points[[*edge]], size=40)

    def plot_delaunay(self, points : np.ndarray = None) -> None:
        if points is not None:
            self.__init__(points)

        plot_data(self.points, color='pink')
        plt.triplot(self.points[:,0], self.points[:,1], self.triangulation.simplices, color='hotpink')
    
    def plot_all(self, points : np.ndarray = None) -> None:
        if points is not None:
            self.__init__(points)

        plot_data(self.points)
        for edge in self.gabriel_edges:
            plt.plot(self.points[[*edge]][:,0], self.points[[*edge]][:,1], '-', color='palegreen')

        plt.triplot(self.points[:,0], self.points[:,1], self.triangulation.simplices, color='deeppink', alpha=0.3)



    
# %%

# %%
