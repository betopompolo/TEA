import matplotlib.pyplot as plt
from kmeans import Clusters, Centroids


def plot(clusters: Clusters, centroids: Centroids, title: str, subtitle: str):
    clusters_colors = ["g", "r", "c", "b", "k"]
    colors_len = len(clusters_colors)
    plt_centroid = None
    plt_point = None

    for index in clusters:
        color = clusters_colors[index % colors_len]
        for point in clusters[index]:
            plt_point = plt.scatter(point[0], point[1], marker=".", color=color)

        plt_centroid = plt.scatter(centroids[index][0], centroids[index][1], marker="x", color=color)

    plt.suptitle(title)
    plt.title(subtitle)
    plt.legend((plt_point, plt_centroid), ('Ponto', 'Centroide'))

    plt.show()
