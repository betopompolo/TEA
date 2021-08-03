import numpy as np

Point = list[float, float]
Clusters = dict[int, list[Point]]


class KMeans:
    def __init__(self, n_clusters=2, max_iter=300, tol=0.001):
        self.n_clusters = n_clusters
        self.centroids: dict[int, Point] = {}
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, dataset: list[Point]):
        self.__init_centroids(dataset)

        for i in range(self.max_iter):
            clusters = self.__create_clusters()

            for data in dataset:
                centroid_dist = [euclidean_dist(data, self.centroids[cluster_index]) for cluster_index in
                                 self.centroids]
                cluster_index = centroid_dist.index(min(centroid_dist))
                clusters[cluster_index].append(data)

            new_centroids = {}
            for cluster_index in clusters:
                new_centroids[cluster_index] = mid_point(clusters[cluster_index])

            is_fit_done = True
            for cluster_index in self.centroids:
                c_dist = euclidean_dist(new_centroids[cluster_index], self.centroids[cluster_index])
                if c_dist > self.tol:
                    is_fit_done = False
                    break

            if is_fit_done:
                break
            else:
                self.centroids = new_centroids

    def predict(self, dataset: list[Point]):
        predictions = []
        for data in dataset:
            distances = [euclidean_dist(data, self.centroids[centroid]) for centroid in self.centroids]
            cluster_index = distances.index(min(distances))
            predictions.append(cluster_index)

        return predictions

    def __init_centroids(self, dataset: list[Point]):
        for i in range(self.n_clusters):
            self.centroids[i] = dataset[i]

    def __create_clusters(self):
        clusters = {}
        for i in range(self.n_clusters):
            clusters[i] = []

        return clusters


def euclidean_dist(p1: Point, p2: Point, abs_dist = True) -> float:
    dist = np.linalg.norm(np.array(p1) - np.array(p2))
    return abs(dist) if abs_dist else dist


def mid_point(points: list[Point]):
    points_count = len(points)
    x = sum([p[0] for p in points]) / points_count
    y = sum([p[1] for p in points]) / points_count

    return [x, y]
