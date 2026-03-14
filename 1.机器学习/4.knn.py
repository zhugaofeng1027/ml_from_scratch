import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X, self.Y = X, y

    def predict(self, X_test):
        preds = []
        for x in X_test:
           dists = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
           idx = np.argsort(dists)[:self.k]
           k_labels = self.y_train[dists]
           label = Counter(k_labels).most_common(1)[0][0]
           preds.append(label)
        return np.array(label)
    
if __name__ == "__main__":
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([[1.5, 2.5], [3.5, 4.5]])

    clf = KNN(k=3)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print("predictions:", predictions)
        
