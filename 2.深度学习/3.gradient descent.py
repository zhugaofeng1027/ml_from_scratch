import numpy as np

def gradient_descent_for_linear_regression(X, y, lr=0.01, epochs=100):
    w = 0.0
    b = 0.0
    m = len(X)
    loss_history = []

    for epoch in range(epochs):
        y_pred = w * X + b
        error = y_pred - y

        dw = (1 / m) * np.dot(error, X)
        db = (1 / m) * np.sum(error)

        w -= lr * dw
        b -= lr * db

        loss = (1 / (2 * m)) * np.sum(error ** 2)
        loss_history.append(loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {loss:.4f}, w {w:.4f}, b {b:.4f}")

    return w, b, loss_history

if __name__ == "__main__":
    np.random.seed(0)
    X = np.linspace(0, 10, 100)
    y = 2 * X + 1 + np.random.randn(100) * 0.5

    w_final, b_final, loss_history = gradient_descent_for_linear_regression(X, y)
    print(f"Final w: {w_final:.4f}, Final b: {b_final:.4f}")

    import matplotlib.pyplot as plt
    # plt.plot(loss_history)
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.title("Loss History")
    # plt.show()
    plt.plot(X, y)
    plt.show()