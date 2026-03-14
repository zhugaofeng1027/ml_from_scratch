import numpy as np

def linear_regression_with_L2(X, y, lr=0.1, lambd=0.1, epochs = 100):
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0

    for _ in range(epochs):
        y_pred = X @ w + b
        error = y_pred - y
        
        dw = (X.T @ error) / m + lambd * w
        db = np.mean(error)

        w -= lr * dw
        b -= lr * db

    return w, b

def linear_regression_with_L1(X, y, lr=0.1, lambd=0.1, epochs=100):
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0

    for _ in range(epochs):
        y_pred = X @ w + b
        error = y_pred - y
        
        dw = (X.T @ error) / m + lambd * np.sign(w)
        db = np.mean(error)

        w -= lr * dw
        b -= lr * db
    return w, b

if __name__ == "__main__":
    np.random.seed(0)
    X = np.random.randn(100, 3)
    true_w = np.array([2.0, -3.0, 1.0])
    y = X @ true_w + np.random.randn(100) * 0.5

    print("--- L2 norm ---")
    w_l2, b_l2 = linear_regression_with_L2(X, y)
    print("w_l2:", w_l2)

    print("--- L1 norm ---")
    w_l1, b_l1 = linear_regression_with_L1(X, y)
    print("w_l1:", w_l1)
