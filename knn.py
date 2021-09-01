# %%
# Librerias utilizadas:
import numpy as np
from typing import Union, List
from scipy.stats import multivariate_normal, bernoulli
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from matplotlib.ticker import FixedLocator, FixedFormatter

import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.cluster import KMeans
from itertools import combinations

from graph import GraphTools
from metrics import dunn
from visualizations import plot_clusters

# %%
# Creación de datos. 
def gaussian_blob(n: int , mean : Union[List,np.ndarray], cov : Union[List, np.ndarray] = None) -> np.ndarray:
    """Generates N-dimensional n gaussian samples using the scipy library. 
       If cov is None it defaults to the identity."""

    norm = multivariate_normal(mean=mean, cov=cov)
    blob = norm.rvs(n)
    return blob

def make_data(n : int = 300, mean_1 = [4,4], mean_2 = [0,0]) -> np.ndarray:
    """Samples from 0.5*N(0,I) and 0.5*N((4,4), I) given a bernoulli trial."""

    bernoulli_trials = np.sum(bernoulli.rvs(0.5, size=n))
    normal_1_trials = bernoulli_trials 
    normal_2_trials = n - bernoulli_trials 

    normal_1_points = 0.5*gaussian_blob(normal_1_trials, mean=mean_2)
    normal_2_points = 0.5*gaussian_blob(normal_2_trials, mean=mean_1) 

    return np.concatenate([normal_1_points, normal_2_points], axis=0)

# %%
if __name__ == '__main__':
    data = make_data()
    n_clusters = 2
    kmeans = KMeans(n_clusters)
    kmeans.fit(data)
    theme = 'dark'

    plot_clusters(data, kmeans, theme=theme)
    plt.show()

# %%
    gt = GraphTools(data, kmeans)
    D = gt.delaunay_graph()
    G = gt.gabriel_graph()

    gt.draw(G, theme=theme)
    plt.show()

# %%
    fig = plt.figure(figsize=(12,12))
    for n_cluster in range(0,len(kmeans.cluster_centers_)):
        nG = gt.get_cluster_subgraph(G, n_cluster)
        tG = gt.get_transitions_subgraph(G, n_cluster)
        fig, plot = gt.draw(tG, fig=fig, shortest_distance = True, theme='light', edge_color='dimgray')
        fig, plot = gt.draw(nG, fig=fig, theme=theme)

    plt.show()
    print('Modified Dunn:', gt.modified_dunn_index(G, n_clusters))
    print('Dunn index:', dunn(data, kmeans.labels_))
    print('Silhouette score:', silhouette_score(data, kmeans.labels_))

# %%
    ## Agregamos los outliers.
    outliers = gaussian_blob(10, [20,20], cov = 0.05*np.array([[1,0], [0,1]]))
    points = np.concatenate([data, outliers])
    n_clusters = 2
    kmeans = KMeans(n_clusters)
    kmeans.fit(points)

    plot_clusters(points, kmeans, theme=theme)
    plt.show()

# %%
    gt = GraphTools(points, kmeans)
    D = gt.delaunay_graph()
    G = gt.gabriel_graph()

    gt.draw(G, theme=theme, node_size=5)
    plt.show()

# %%
    fig = plt.figure(figsize=(12,12))
    for n_cluster in range(0,len(kmeans.cluster_centers_)):
        nG = gt.get_cluster_subgraph(G, n_cluster)
        tG = gt.get_transitions_subgraph(G, n_cluster)
        fig, plot = gt.draw(tG, fig=fig, shortest_distance = True, theme='light', edge_color='dimgray')
        fig, plot = gt.draw(nG, fig=fig, theme=theme)

    plt.show()
    print('Modified Dunn:', gt.modified_dunn_index(G, n_clusters))
    print('Dunn index: ', dunn(points, kmeans.labels_))
    print('Silhouette score:', silhouette_score(points, kmeans.labels_))

# %%
    data = make_data(mean_1=[1.3, 1.3])
    n_clusters = 2
    kmeans = KMeans(n_clusters)
    kmeans.fit(data)
    theme = 'dark'

    plot_clusters(data, kmeans, theme=theme)
    plt.show()

# %%
    gt = GraphTools(data, kmeans)
    D = gt.delaunay_graph()
    G = gt.gabriel_graph()

    gt.draw(G, theme=theme)
    plt.show()

# %%
    fig = plt.figure(figsize=(12,12))
    for n_cluster in range(0,len(kmeans.cluster_centers_)):
        nG = gt.get_cluster_subgraph(G, n_cluster)
        tG = gt.get_transitions_subgraph(G, n_cluster)
        fig, plot = gt.draw(tG, fig=fig, shortest_distance = True, theme='light', edge_color='dimgray')
        fig, plot = gt.draw(nG, fig=fig, theme=theme)

    plt.show()
    print('Modified Dunn:', gt.modified_dunn_index(G, n_clusters))
    labels = kmeans.predict(data)
    print('Dunn index: ', dunn(data, labels))
    print('Silhouette score:', silhouette_score(data, kmeans.labels_))

# %%
    ## Agregamos los outliers.
    outliers = gaussian_blob(10, [20,20], cov = 0.05*np.array([[1,0], [0,1]]))
    points = np.concatenate([data, outliers])
    n_clusters = 2
    kmeans = KMeans(n_clusters)
    kmeans.fit(points)

    plot_clusters(points, kmeans, theme=theme)
    plt.show()

# %%
    gt = GraphTools(points, kmeans)
    D = gt.delaunay_graph()
    G = gt.gabriel_graph()

    gt.draw(G, theme=theme, node_size=5)
    plt.show()

# %%
    fig = plt.figure(figsize=(12,12))
    for n_cluster in range(0,len(kmeans.cluster_centers_)):
        nG = gt.get_cluster_subgraph(G, n_cluster)
        tG = gt.get_transitions_subgraph(G, n_cluster)
        fig, plot = gt.draw(tG, fig=fig, shortest_distance = True, theme='light', edge_color='dimgray')
        fig, plot = gt.draw(nG, fig=fig, theme=theme)

    plt.show()
    print('Modified Dunn:', gt.modified_dunn_index(G, n_clusters))
    print('Dunn index: ', dunn(points, kmeans.labels_))
    print('Silhouette score:', silhouette_score(points, kmeans.labels_))

# %%
    ## Elbow plot:
    data = make_data()
    kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(data)
                    for k in range(1, 10)]
    inertias = [model.inertia_ for model in kmeans_per_k]

    plt.figure(figsize=(14, 7))
    plt.plot(range(1, 10), inertias, "o-", c = 'teal', lw=4, ms=14)
    plt.xlabel("$k$", fontsize=16)
    plt.ylabel("W", fontsize=16)
    plt.ylim(0, max(inertias) + 10)
    plt.tick_params(labelsize = 14)
    plt.annotate('Elbow',
                xy=(2, inertias[1]),
                xytext=(0.25, 0.55),
                textcoords='figure fraction',
                fontsize=16,
                arrowprops=dict(facecolor='black', shrink=0.1)
                )
    plt.grid(True,c='darkgrey', alpha=0.3)
    plt.show()

# %%
    ## Silhouette knives:
    plt.figure(figsize=(16, 12))
    silhouette_scores = [silhouette_score(data, model.labels_)
                        for model in kmeans_per_k[1:]]

    for k in (2, 3, 4, 5):
        plt.subplot(2, 2, k - 1)
        
        y_pred = kmeans_per_k[k - 1].labels_
        silhouette_coefficients = silhouette_samples(data, y_pred)

        padding = len(data) // 30
        pos = padding
        ticks = []
        for i in range(k):
            coeffs = silhouette_coefficients[y_pred == i]
            coeffs.sort()

            color = mpl.cm.Spectral(i / k)
            plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs,
                            facecolor=color, edgecolor=color, alpha=0.7)
            ticks.append(pos + len(coeffs) // 2)
            pos += len(coeffs) + padding

        plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
        plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))
        if k in (2, 4):
            plt.ylabel("Cluster", fontsize=16)
        
        if k in (4, 5):
            plt.xlabel("Silhouette Coefficient", fontsize=16)

        plt.tick_params(labelsize = 14)
        plt.axvline(x=silhouette_scores[k - 1], color="red", linestyle="--")
        plt.title("$k={}$".format(k), fontsize=16)
        plt.grid(True,c='darkgrey', alpha=0.3)

    plt.show()

# %%
    kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(data)
                    for k in range(2, 10)]
    silhouette_scores = [silhouette_score(data, model.labels_) for model in kmeans_per_k]
    dunn_scores =  [dunn(data, model.labels_) for model in kmeans_per_k]

    modified_dunn = []
    for k in range(2,10):
        kmeans = KMeans(k)
        kmeans.fit(data)
        gt = GraphTools(data, kmeans)
        G = gt.gabriel_graph()
        modified_dunn.append(gt.modified_dunn_index(G, k))

    plt.figure(figsize=(14, 7))
    # plt.plot(range(2, 10), silhouette_scores, "o-", c = 'darkorchid', lw=2, ms=9, label = 'Silhouette')
    plt.plot(range(2, 10), dunn_scores, "o-", c = 'teal', lw=2, ms=9, label='Dunn')
    plt.plot(range(2, 10), modified_dunn, "o-", c = 'tomato', lw=2, ms=9, label='Modified Dunn')
    plt.xlabel("$k$", fontsize=16)
    plt.ylabel("Score", fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True,c='darkgrey', alpha=0.3)
    plt.show()

# %%
    #Silhouette scores for different k.

    plt.figure(figsize=(14, 7))
    plt.plot(range(2, 10), silhouette_scores, "o-", c = 'darkorchid', lw=2, ms=9, label = 'Silhouette')
    plt.xlabel("$k$", fontsize=16)
    plt.ylabel("Score", fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True,c='darkgrey', alpha=0.3)
    plt.show()

# %%
