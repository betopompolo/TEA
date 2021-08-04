import matplotlib.pyplot as plt
from kmeans import Clusters, Centroids


def plot(clusters: Clusters, centroids: Centroids):
    clusters_colors = ["g", "r", "c", "b", "k"]
    colors_len = len(clusters_colors)
    plt_centroid = None
    plt_point = None

    for index in clusters:
        color = clusters_colors[index % colors_len]
        plt_point = plt.scatter(centroids[index][0], centroids[index][1], marker="x", color=color)
        for point in clusters[index]:
            plt_centroid = plt.scatter(point[0], point[1], marker="o", color=color)

    plt.title('K-Means')
    plt.legend((plt_point, plt_centroid), ('Ponto', 'Centroide'))

    plt.show()
