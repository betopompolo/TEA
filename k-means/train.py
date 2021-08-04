import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer

from kmeans import KMeans
from plot import plot


def run_iris():
    dataset = load_iris()
    features = [0, 1]
    X = get_features(dataset.data, features)
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    predicts = kmeans.predict(X)
    plot(predicts, kmeans.centroids, title='Iris dataset',
         subtitle=' x '.join(dataset.target_names[f] for f in features))


def run_wine():
    dataset = load_wine()
    features = [0, 1]
    X = get_features(dataset.data, features)
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    predicts = kmeans.predict(X)
    plot(predicts, kmeans.centroids, title='Wine dataset',
         subtitle=' x '.join(dataset.target_names[f] for f in features))


def run_breast_cancer():
    dataset = load_breast_cancer()
    features = [0, 1]
    X = get_features(dataset.data, features)
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    predicts = kmeans.predict(X)
    plot(predicts, kmeans.centroids, title='Breast cancer dataset',
         subtitle=' x '.join(dataset.target_names[f] for f in features))


def get_features(data: np.ndarray, indexes: list[int]):
    d0 = data[:, indexes[0]]
    d1 = data[:, indexes[1]]

    return [[p0, p1] for p0, p1 in zip(d0, d1)]
