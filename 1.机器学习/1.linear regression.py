import numpy as np

# def compute_cost(X, y, theta):
#     m = len(y)
#     predictions = X.dot(theta)
#     cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
#     return cost

def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        predictions = X.dot(theta)
        error = predictions - y
        gradient = (1 / m) * X.T.dot(error)
        theta = theta - alpha * gradient
        # J_history[i] = compute_cost(X, y, theta)
    return theta, J_history

if __name__ == "__main__":
    np.random.seed(0)
    
    X = 2 * np.random.rand(100, 1)
    # print(X, X.shape)
    y = 4 + 3 * X + np.random.randn(100, 1)
    # print(y, y.shape)

    X_b = np.c_[np.ones((100, 1)), X]
    # print(X_b, X_b.shape)
    theta = np.random.randn(2,1)
    # print(theta, theta.shape)

    alpha = 0.01
    num_iters = 1000

    final_theta, cost_history = gradient_descent(X_b, y, theta, alpha, num_iters)

    print("---线形回归(SGD)---")
    print("优化后的参数(theta):")
    print(final_theta)
    print("梯度更新的记录(loss)")
    print(cost_history)
