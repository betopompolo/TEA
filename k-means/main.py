from kmeans import KMeans
from plot import plot

X = [[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]]
y = [[10, 3], [1, 1]]
k = 2

kmeans = KMeans()
kmeans.fit(X)
predicts = kmeans.predict(y)
plot(predicts, kmeans.centroids)
