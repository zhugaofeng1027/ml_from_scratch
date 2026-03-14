import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(linear_model)
            
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_prob(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_model)
        return y_pred
    
    def predict(self, X, threshold=0.5):
        y_pred_prob = self.predict_prob(X)
        # y_pred = np.where(y_pred_prob >= 0.5, 1, 0)
        y_pred = [1 if i > threshold else 0 for i in y_pred_prob]
        return y_pred
            
if __name__ == "__main__":
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([0, 0, 1, 1])
    model = LogisticRegression()
    model.fit(X, y)
    y_pred_prob = model.predict_prob(X)
    print(y_pred_prob)
    y_pred = model.predict(X)
    print(y_pred)