import numpy as np


class MyPCA:
    def __init__(self, n_components: int):
        self.n_components = n_components

    def transform(self, x):
        x_centered = x - np.mean(x, axis=0)
        covariance_matrix = np.cov(x_centered, rowvar=False)

        eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)
        sorted_index = np.argsort(eigen_values)[::-1]
        sorted_eigenvectors = eigen_vectors[:, sorted_index]
        feature_vectors = sorted_eigenvectors[:, 0:self.n_components]

        result = np.dot(feature_vectors.transpose(), x_centered.transpose()).transpose()
        return result


class MyLDA:
    def __init__(self, n_components: int):
        self.n_components = n_components

    def transform(self, x: np.ndarray, y: np.ndarray):
        height, width = x.shape
        unique_classes = np.unique(y)
        num_classes = len(unique_classes)

        scatter_t = np.cov(x.T) * (height - 1)
        scatter_w = 0
        for i in range(num_classes):
            class_items = np.flatnonzero(y == unique_classes[i])
            scatter_w = scatter_w + np.cov(x[class_items].T) * (len(class_items) - 1)

        scatter_b = scatter_t - scatter_w
        eig_vectors = np.linalg.eigh(np.linalg.pinv(scatter_w).dot(scatter_b))[1]
        feature_vector = eig_vectors[:, ::-1][:, :self.n_components]
        result = np.dot(x, feature_vector)

        return result
