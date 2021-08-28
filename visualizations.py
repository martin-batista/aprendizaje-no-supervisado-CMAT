from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import matplotlib
import matplotlib.pyplot as plt
from typing import Callable, List
from itertools import cycle
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np

def make_plot(fig = None) -> None:
    """Plot templates."""
    if fig is None:
        fig = plt.figure(figsize=(12,12))
        plt.style.use('dark_background')
        plt.rc('axes',edgecolor='k')
        plt.grid(True,c='darkgrey', alpha=0.3)
    else: 
        plt.rcParams["figure.figsize"] = (12,12)
        plt.style.use('dark_background')
        plt.rc('axes',edgecolor='k')
        plt.grid(True,c='darkgrey', alpha=0.3)
    
    ax = fig.gca()
    ax.tick_params(axis='x', colors='darkgrey')
    ax.tick_params(axis='y', colors='darkgrey')

    return plt

def make_cmap(colors : List = None) -> None:
    """Creates a custom linear colormap."""
    if colors is None:
        colors = ["hotpink", "orchid", "palegreen", "mediumspringgreen", "aqua", "dodgerblue"]
    cmap = LinearSegmentedColormap.from_list("custom_bright", colors)
    return cmap
 
def plot_data(data : np.ndarray, plt : matplotlib.pyplot = None, y_labels : np.ndarray = None, color : str  = None, 
              cmap : str = None, size : int = 15, alpha : float = 1) -> None:
    """Plots the data points. Colors them based on the y_labels parameter if available."""
    if plt is None:
        plt = make_plot()
    
    if cmap is None:
        cmap = make_cmap()

    if y_labels is None and color is None:
        color = 'cyan'
    elif y_labels is not None:
        color = y_labels

    plt.scatter(data[:,0], data[:,1], c = color, marker='o', cmap=cmap, s=size, zorder= 14, alpha=alpha)
    plt.scatter(data[:,0], data[:,1], color='k', marker='o', cmap=cmap, s=size, zorder=1, alpha=0.6, linewidths=6)
    plt.grid(True,c='darkgrey', alpha=0.3)
    return plt

def plot_centroids(centroids : np.ndarray, center_labels : np.ndarray, plt : matplotlib.pyplot, cmap: str,weights : float =None) -> None:
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=85, linewidths=12,
                c=center_labels, cmap = cmap, zorder=20, alpha=0.4)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=5, linewidths=17,
                color='k', cmap = cmap, zorder=25, alpha=1)
    return plt

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def plot_clusters(X : np.ndarray, clusterer : Callable,  show_boundaries : bool = True, plt : matplotlib.pyplot = None,
                 cmap : str = None, resolution : int =1000, show_centroids : bool =True,
                 show_xlabels : bool =True, show_ylabels :bool =True,) -> None:
    """Plots the cluster data and the decision boundaries. If the number of clusters is above 4, it utilizes
       the Voronoi tessellation to construct the boundaries.
    """
    # Plot the data points and cluster centers.
    plt = make_plot(plt)
    if cmap is None:
        cmap = make_cmap()
    cluster_labels = clusterer.predict(X)
    center_labels = clusterer.predict(clusterer.cluster_centers_)
    plt = plot_data(X, plt, cluster_labels)

    if show_centroids:
        plt = plot_centroids(clusterer.cluster_centers_, center_labels, plt,cmap)
    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)
    
    # Plot the boundaries.
    if show_boundaries:
        # If the number of clusters is below 4, plot the contour and fill for the cluster centers.
        if len(clusterer.cluster_centers_) <= 9:
            mins = X.min(axis=0) - 0.1
            maxs = X.max(axis=0) + 0.1

            xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                                np.linspace(mins[1], maxs[1], resolution))
            Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                        cmap=cmap, alpha=0.2)
            plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                        linewidths=1, colors='k')

        else:
            # If the number of clusters exceeds 4, use the Voronoi tessellation instead.
            cluster_labels = clusterer.predict(X)
            lst = [i for i, _ in enumerate(clusterer.cluster_centers_)]
            minima = lst[0]
            maxima = lst[-1]
            color_cycler = cycle(lst) # Constructs a cycle iterator to pass on as a color for the regions.

            norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
            mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

            vor = Voronoi(clusterer.cluster_centers_)

            regions, vertices = voronoi_finite_polygons_2d(vor)
            for region in regions:
                rgba_value = next(color_cycler)
                polygon = vertices[region]
                plt.fill(*zip(*polygon),  color=mapper.to_rgba(rgba_value), alpha=0.2)

            plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
            plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)

            voronoi_plot_2d(vor, plt.gca(), show_vertices=False, line_colors="black", line_width=1.5, point_size=0)
        return plt
