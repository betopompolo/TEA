from kmeans import KMeans


X = [[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]]
y = [[10, 3], [1, 1]]

kmeans = KMeans()
kmeans.fit(X)
predicts = kmeans.predict(y)
print(predicts)
