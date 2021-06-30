import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

from analysis import MyLDA, MyPCA


def plot(x, y, target_names, title: str):
    colors = ['navy', 'turquoise', 'darkorange']
    lw = 2

    plt.figure()
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(x[y == i, 0], x[y == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title(title)


if __name__ == '__main__':
    iris = datasets.load_iris()
    X, target, names = iris.data, iris.target, iris.target_names
    n_components = [2, 3]

    for n in n_components:
        lda = MyLDA(n)
        lda_matrix = np.array(lda.transform(X, target))
        plot(lda_matrix, target, names, f'LDA (nº componentes: {n})')

        pca = MyPCA(n)
        pca_matrix = np.array(pca.transform(X))
        plot(pca_matrix, target, names, f'PCA (nº componentes: {n})')

    plt.show()
