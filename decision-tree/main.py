from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    dataset = load_iris()
    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    classifiers = [
        RandomForestClassifier(max_depth=2, random_state=0),
        tree.DecisionTreeClassifier(),
        GradientBoostingClassifier()
    ]
    for clf in classifiers:
        clf.fit(X_train, y_train)
        print(f'{clf} score: {clf.score(X_test, y_test)}')
